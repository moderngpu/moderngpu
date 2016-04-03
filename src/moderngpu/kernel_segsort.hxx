// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once

#include "search.hxx"
#include "cta_segsort.hxx"
#include "cta_scan.hxx"

BEGIN_MGPU_NAMESPACE

namespace detail {

template<typename launch_arg_t, typename key_t, typename val_t, 
  typename comp_t>
struct segsort_t {
  enum { has_values = !std::is_same<val_t, empty_t>::value };
  typedef typename conditional_typedef_t<launch_arg_t, 
    launch_box_t<
      arch_20_cta<128, 15>,
      arch_35_cta<128, 11>,
      arch_52_cta<128, 15>
    >
  >::type_t launch_t;

  context_t& context;
  comp_t comp;
  cta_dim_t cta_dim;
  int count, nv, num_ctas, num_passes;

  mem_t<key_t> keys_temp;
  mem_t<val_t> vals_temp;

  key_t* keys_source, *keys_dest, *keys_blocksort;
  val_t* vals_source, *vals_dest, *vals_blocksort;

  mem_t<range_t> merge_ranges;
  mem_t<merge_range_t> merge_list;
  mem_t<int> compressed_ranges, copy_list, copy_status;
  mem_t<int2> op_counters;

  segsort_t(key_t* keys, val_t* vals, int count_, comp_t comp_, 
    context_t& context_) : count(count_), comp(comp_), context(context_) { 

    nv = launch_t::nv(context);
    num_ctas = div_up(count, nv);
    num_passes = find_log2(num_ctas, true);
    
    int capacity = num_ctas;                 // log(num_ctas) per pass.
    for(int i = 0; i < num_passes; ++i)
      capacity += div_up(num_ctas, 1<< i);

    if(num_passes              ) keys_temp = mem_t<key_t>(count, context);
    if(num_passes && has_values) vals_temp = mem_t<val_t>(count, context);

    keys_source = keys;
    vals_source = vals;
    keys_dest = keys_temp.data();
    vals_dest = vals_temp.data();

    // The blocksort passes outputs to these arrays.
    keys_blocksort = (1 & num_passes) ? keys_dest : keys_source;
    vals_blocksort = (1 & num_passes) ? vals_dest : vals_source;

    // Allocate space for temporary variables.
    merge_ranges = mem_t<range_t>(capacity, context);
    merge_list = mem_t<merge_range_t>(num_ctas, context);
    compressed_ranges = mem_t<int>(num_ctas, context);
    copy_list = mem_t<int>(num_ctas, context);
    copy_status = mem_t<int>(num_ctas, context);
    op_counters = fill<int2>(int2(), num_passes, context);
  }

  template<bool sort_indices = false, typename keys_it, typename vals_it, 
    typename segments_it>
  void blocksort_segments(keys_it keys, vals_it vals, segments_it segments, 
    int num_segments) {

    // Distribute the segment descriptors to different CTAs.
    mem_t<int> partitions = binary_search_partitions<bounds_lower>(segments, 
      count, num_segments, nv, context);
    const int* mp_data = partitions.data();

    ////////////////////////////////////////////////////////////////////////////
    // Block sort the input. The position of the first and last segment 
    // descriptors are stored to merge_ranges.

    int count = this->count;
    key_t* keys_blocksort = this->keys_blocksort;
    val_t* vals_blocksort = this->vals_blocksort;
    int* compressed_ranges_data = compressed_ranges.data();

    auto blocksort_k = [=] MGPU_DEVICE(int tid, int cta) {
      typedef typename launch_t::sm_ptx params_t;
      enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };
      typedef cta_load_head_flags<nt, vt> load_head_flags_t;
      typedef cta_segsort_t<nt, vt, key_t, val_t> sort_t;

      __shared__ union {
        typename load_head_flags_t::storage_t load_head_flags;
        typename sort_t::storage_t sort;
        key_t keys[nv + 1];
        val_t vals[nv];
      } shared;

      // Load the partitions for the segment descriptors and extract head 
      // flags for each key.
      int p[2] = { mp_data[cta], mp_data[cta + 1] };
      int head_flags = load_head_flags_t().load(segments, p, tid, cta, 
        count, shared.load_head_flags);

      // Load the keys and values.
      range_t tile = get_tile(cta, nv, count);

      kv_array_t<key_t, val_t, vt> unsorted;
      unsorted.keys = mem_to_reg_thread<nt, vt>(keys + tile.begin, tid, 
        tile.count(), shared.keys);
      if(sort_indices) {
        // If we're sorting indices, load from the counting_iterator_t directly
        // without staging through shared memory.
        iterate<vt>([&](int i) {
          unsorted.vals[i] = vals[tile.begin + vt * tid + i];
        });
      } else if(has_values) {
        // If we're storing actual values, stage through shared memory.
        unsorted.vals = mem_to_reg_thread<nt, vt>(vals + tile.begin, tid,
          tile.count(), shared.vals);
      }

      // Blocksort.
      range_t active;
      kv_array_t<key_t, val_t, vt> sorted = sort_t().block_sort(unsorted,
        tid, tile.count(), head_flags, active, comp, shared.sort);

      // Store the keys and values.
      reg_to_mem_thread<nt, vt>(sorted.keys, tid, tile.count(), 
        keys_blocksort + tile.begin, shared.keys);
      if(has_values)
        reg_to_mem_thread<nt, vt>(sorted.vals, tid, tile.count(), 
          vals_blocksort + tile.begin, shared.vals);

      // Store the active range for the entire CTA. These are used by the 
      // segmented partitioning kernels.
      if(!tid)
        compressed_ranges_data[cta] = bfi(active.end, active.begin, 16, 16);
    };
    cta_transform<launch_t>(blocksort_k, count, context);

    if(1 & num_passes) {
      std::swap(this->keys_source, this->keys_dest);
      std::swap(this->vals_source, this->vals_dest);
    }
  }

  void merge_passes() {

    ////////////////////////////////////////////////////////////////////////////
    // Execute a partitioning and a merge for each mergesort pass.

    int num_ranges = num_ctas;
    int num_partitions = num_ctas + 1;
    int count = this->count;
    int nv = this->nv;

    key_t* keys_source = this->keys_source;
    val_t* vals_source = this->vals_source;
    key_t* keys_dest = this->keys_dest;
    val_t* vals_dest = this->vals_dest;

    range_t* source_ranges = merge_ranges.data();
    range_t* dest_ranges = merge_ranges.data();

    const int* compressed_ranges_data = compressed_ranges.data();
    int* copy_status_data = copy_status.data();
    int* copy_list_data = copy_list.data();
    merge_range_t* merge_list_data = merge_list.data();
    int2* op_counters_data = op_counters.data();

    for(int pass = 0; pass < num_passes; ++pass) {
      int coop = 2<< pass;

      //////////////////////////////////////////////////////////////////////////
      // Partition the data within its segmented mergesort list.

      enum { nt = 64 };
      int num_partition_ctas = div_up(num_partitions, nt - 1);

      auto partition_k = [=] MGPU_DEVICE(int tid, int cta) {
        typedef cta_scan_t<nt, int> scan_t;
        __shared__ union {
          typename scan_t::storage_t scan;
          int partitions[nt + 1];
          struct { int merge_offset, copy_offset; };
        } shared;

        int partition = (nt - 1) * cta + tid;
        int first = nv * partition;
        int count2 = min(nv, count - first);

        int mp0 = 0;
        bool active = (tid < nt - 1) && (partition < num_partitions - 1);
        int range_index = partition>> pass;

        if(partition < num_partitions) {

          merge_range_t range = compute_mergesort_range(count, partition, 
            coop, nv);
          int diag = min(nv * partition - range.a_begin, range.total());

          int indices[2] = { 
            min(num_ranges - 1, ~1 & range_index), 
            min(num_ranges - 1, 1 | range_index) 
          };
          range_t ranges[2];

          if(pass > 0) {
            ranges[0] = source_ranges[indices[0]];
            ranges[1] = source_ranges[indices[1]];
          } else {
            iterate<2>([&](int i) {
              int compressed = compressed_ranges_data[indices[i]];
              int first = nv * indices[i];

              ranges[i] = range_t { 0x0000ffff & compressed, compressed>> 16 };
              if(nv != ranges[i].begin) ranges[i].begin += first;
              else ranges[i].begin = count;
              if(-1 != ranges[i].end) ranges[i].end += first;
            });
          }

          range_t inner = { 
            ranges[0].end, 
            max(range.b_begin, ranges[1].begin) 
          };
          range_t outer = { 
            min(ranges[0].begin, ranges[1].begin),
            max(ranges[0].end, ranges[1].end)
          };

          // Segmented merge path on inner.
          mp0 = segmented_merge_path(keys_source, range, inner, diag, comp);

          // Store outer merge range.
          if(active && 0 == diag)
            dest_ranges[range_index / 2] = outer;
        }
        shared.partitions[tid] = mp0;
        __syncthreads();

        int mp1 = shared.partitions[tid + 1];
        __syncthreads();

        // Update the merge range to include partitioning.
        merge_range_t range = compute_mergesort_range(count, partition, coop, 
          nv, mp0, mp1);

        // Merge if the source interval does not exactly cover the destination
        // interval. Otherwise copy or skip.
        range_t interval = (1 & range_index) ? 
          range.b_range() : range.a_range();
        bool merge_op = false;
        bool copy_op = false;

        // Create a segsort job.
        if(active) {
          merge_op = (first != interval.begin) || (interval.count() != count2);
          copy_op = !merge_op && (!pass || !copy_status_data[partition]);

          // Use the b_end component to store the index of the destination tile.
          // The actual b_end can be inferred from a_count and the length of 
          // the input array.
          range.b_end = partition;
        }

        // Scan the counts of merges and copies.
        scan_result_t<int> merge_scan = scan_t().scan(tid, (int)merge_op, 
          shared.scan);
        scan_result_t<int> copy_scan = scan_t().scan(tid, (int)copy_op, 
          shared.scan);

        // Increment the operation counters by the totals.
        if(!tid) {
          shared.merge_offset = atomicAdd(&op_counters_data[pass].x, 
            merge_scan.reduction);
          shared.copy_offset = atomicAdd(&op_counters_data[pass].y, 
            copy_scan.reduction);
        }
        __syncthreads();

        if(active) {
          copy_status_data[partition] = !merge_op;
          if(merge_op)
            merge_list_data[shared.merge_offset + merge_scan.scan] = range;
          if(copy_op)
            copy_list_data[shared.copy_offset + copy_scan.scan] = partition;
        }
      };
      cta_launch<nt>(partition_k, num_partition_ctas, context);

      source_ranges = dest_ranges;
      num_ranges = div_up(num_ranges, 2);
      dest_ranges += num_ranges;

      //////////////////////////////////////////////////////////////////////////
      // Merge or copy unsorted tiles.

      auto merge_k = [=] MGPU_DEVICE(int tid, int cta) {
        typedef typename launch_t::sm_ptx params_t;
        enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };

        __shared__ union {
          key_t keys[nv + 1];
          int indices[nv];
        } shared;

        merge_range_t range = merge_list_data[cta];

        int tile = range.b_end;
        int first = nv * tile;
        int count2 = min((int)nv, count - first);
        range.b_end = range.b_begin + (count2 - range.a_count());

        int compressed_range = compressed_ranges_data[tile];
        range_t active = {
          0x0000ffff & compressed_range,
          compressed_range>> 16
        };
        load_two_streams_shared<nt, vt>(keys_source + range.a_begin, 
          range.a_count(), keys_source + range.b_begin, range.b_count(),
          tid, shared.keys);

        // Run a merge path search to find the starting point for each thread
        // to merge. If the entire warp fits into the already-sorted segments,
        // we can skip sorting it and leave its keys in shared memory.
        int list_parity = 1 & (tile>> pass);
        if(list_parity) active = range_t { 0, active.begin };
        else active = range_t { active.end, nv };

        int warp_offset = vt * (~(warp_size - 1) & tid);
        bool sort_warp = list_parity ?
          (warp_offset < active.end) : 
          (warp_offset + vt * warp_size >= active.begin);
   
        merge_pair_t<key_t, vt> merge;
        merge_range_t local_range = range.to_local();
        if(sort_warp) {
          int diag = vt * tid;
          int mp = segmented_merge_path(shared.keys, local_range,
            active, diag, comp);

          merge_range_t partitioned = local_range.partition(mp, diag);
          merge = segmented_serial_merge<vt>(shared.keys, 
            local_range.partition(mp, diag), active, comp, false);
        } else {
          iterate<vt>([&](int i) {
            merge.indices[i] = vt * tid + i;
          });
        }
        __syncthreads();

        // Store keys to global memory.
        if(sort_warp)
          reg_to_shared_thread<nt, vt>(merge.keys, tid, shared.keys, false);
        __syncthreads();

        shared_to_mem<nt, vt>(shared.keys, tid, count2, keys_dest + first);

        if(has_values) {
          // Transpose the indices from thread order to strided order.
          array_t<int, vt> indices = reg_thread_to_strided<nt>(merge.indices,
            tid, shared.indices);

          // Gather the input values and merge into the output values.
          transfer_two_streams_strided<nt>(vals_source + range.a_begin, 
            range.a_count(), vals_source + range.b_begin, range.b_count(), 
            indices, tid, vals_dest + first);
        }
      };
      cta_launch<launch_t>(merge_k, &op_counters_data[pass].x, context);

      auto copy_k = [=] MGPU_DEVICE(int tid, int cta) {
        typedef typename launch_t::sm_ptx params_t;
        enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };

        int tile = copy_list_data[cta];
        int first = nv * tile;
        int count2 = min((int)nv, count - first);

        mem_to_mem<nt, vt>(keys_source + first, tid, count2, 
          keys_dest + first);

        if(has_values)
          mem_to_mem<nt, vt>(vals_source + first, tid, count2, 
            vals_dest + first);
      };
      cta_launch<launch_t>(copy_k, &op_counters_data[pass].y, context);

      std::swap(keys_source, keys_dest);
      std::swap(vals_source, vals_dest);
    }    
  }
};

} // namespace detail

// Key-value mergesort.
template<typename launch_arg_t = empty_t, typename key_t, typename val_t,
  typename seg_it, typename comp_t>
void segmented_sort(key_t* keys, val_t* vals, int count, seg_it segments, 
  int num_segments, comp_t comp, context_t& context) {

  detail::segsort_t<launch_arg_t, key_t, val_t, comp_t> 
    segsort(keys, vals, count, comp, context);

  segsort.blocksort_segments(keys, vals, segments, num_segments);
  segsort.merge_passes();
}

// Key-value mergesort. Automatically generate indices to sort as values.
template<typename launch_arg_t = empty_t, typename key_t, typename seg_it, 
  typename comp_t>
void segmented_sort_indices(key_t* keys, int* indices, int count, 
  seg_it segments, int num_segments, comp_t comp, context_t& context) {

  detail::segsort_t<launch_arg_t, key_t, int, comp_t> 
    segsort(keys, indices, count, comp, context);

  segsort.template blocksort_segments<true>(keys, counting_iterator_t<int>(), 
    segments, num_segments);
  segsort.merge_passes();
}

// Key-only segmented sort
template<typename launch_arg_t = empty_t, typename key_t, typename seg_it, 
  typename comp_t>
void segmented_sort(key_t* keys, int count, seg_it segments, 
  int num_segments, comp_t comp, context_t& context) {

  segmented_sort<launch_arg_t>(keys, (empty_t*)nullptr, count,
    segments, num_segments, comp, context);
}


END_MGPU_NAMESPACE
