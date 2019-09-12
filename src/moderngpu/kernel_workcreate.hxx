// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once

#include "search.hxx"
#include "cta_load_balance.hxx"
#include "kernel_scan.hxx"
#include "tuple.hxx"

BEGIN_MGPU_NAMESPACE

// experimental feature
namespace expt {

template<typename launch_arg_t, typename segments_it>
struct workcreate_t {
  typedef typename conditional_typedef_t<launch_arg_t, 
    launch_box_t<
      arch_20_cta<128, 11, 8>,
      arch_35_cta<128,  7, 5>,
      arch_52_cta<128, 11, 8>
    >
  >::type_t launch_t;

  segments_it segments;
  int num_segments;
  int count;
  context_t& context;

  cta_dim_t cta_dim;
  int num_ctas;

  mem_t<int> mp;
  mem_t<short> bits;
  mem_t<int2> cta_offsets;
  int2 cta_total;

  struct add_int2_t {
    MGPU_HOST_DEVICE int2 operator()(int2 a, int2 b) const {
      return make_int2(a.x + b.x, a.y + b.y);
    }
  };

public:

  struct count_t {
    int count;
    int num_segments;
  };

  workcreate_t(int count_, segments_it segments_, int num_segments_,
    context_t& context_) : 
    count(count_), segments(segments_), num_segments(num_segments_),
    context(context_) {

    // Compute the number of CTAs.
    cta_dim = launch_t::cta_dim(context);
    num_ctas = cta_dim.num_ctas(count + num_segments);

    mp = load_balance_partitions(count, segments, num_segments, cta_dim.nv(), 
      context);

    bits = mem_t<short>(num_ctas * cta_dim.nt, context);

    cta_offsets = mem_t<int2>(num_ctas, context);
  }

  // f(int index, int seg, int rank, tuple<...> desc) returns the number
  // of work-items to create.
  template<typename func_t, typename pointers_t>
  count_t upsweep(func_t f, pointers_t caching_iterators) {
    
    const int* mp_data = mp.data();
    short* bits_data = bits.data();
    int2* counts_data = cta_offsets.data();
    int count = this->count;
    auto segments = this->segments;
    int num_segments = this->num_segments;

    typedef tuple_iterator_value_t<pointers_t> value_t;
    auto upsweep_k = [=]MGPU_DEVICE(int tid, int cta) {
      typedef typename launch_t::sm_ptx params_t;
      enum { nt = params_t::nt, vt = params_t::vt, vt0 = params_t::vt0 };
      typedef cta_reduce_t<nt, int2> reduce_t;
      typedef cta_load_balance_t<nt, vt> load_balance_t;
      typedef detail::cached_segment_load_t<nt, pointers_t> cached_load_t;

      static_assert(vt <= 16, "mgpu::workcreate_t vt must be <= 16.");

      __shared__ union {
        typename reduce_t::storage_t reduce;
        typename load_balance_t::storage_t lbs;
        typename cached_load_t::storage_t cached;
      } shared;

      // Compute the load-balancing search and materialize (index, seg, rank) 
      // arrays.
      auto lbs = load_balance_t().load_balance(count, segments, num_segments,
        tid, cta, mp_data, shared.lbs);

      // Call the user-supplied functor f.
      short segment_bits = 0;
      int work_items = 0;

      // Load from the cached iterators. Use the placement range, not the 
      // merge-path range for situating the segments.
      array_t<value_t, vt> cached_values = cached_load_t::template load<vt0>(
        tid, lbs.merge_range.a_count(), lbs.placement.range.b_range(), 
        lbs.segments, shared.cached, caching_iterators);
      
      strided_iterate<nt, vt, vt0>([&](int i, int j) {
        int index = lbs.merge_range.a_begin + j;
        int seg = lbs.segments[i];
        int rank = lbs.ranks[i];

        int work_count = f(index, seg, rank, cached_values[i]);

        if(work_count > 0) segment_bits |= 1<< i;
        work_items += work_count;
      }, tid, lbs.merge_range.a_count());

      // Store the worker bits for this thread.
      bits_data[nt * cta + tid] = segment_bits;

      // Scan the segment and work-item counts.
      int2 reduction = reduce_t().reduce(tid, 
        make_int2(popc(segment_bits), work_items), shared.reduce,
        nt, add_int2_t(), false);
      if(!tid) counts_data[cta] = reduction;
    };
    cta_launch<launch_t>(upsweep_k, num_ctas, context);

    // Scan the partial reductions.
    mem_t<int2> counts_host(1, context, memory_space_host);
    scan_event(counts_data, num_ctas, counts_data, add_int2_t(),
      counts_host.data(), context, context.event());
    cudaEventSynchronize(context.event());  

    cta_total = counts_host.data()[0];
    return count_t { cta_total.y, cta_total.x };
  }

  // upsweep without caching iterators.
  template<typename func_t>
  count_t upsweep(func_t f) {
    return upsweep(
      [=]MGPU_DEVICE(int index, int seg, int rank, tuple<>) {
        return f(index, seg, rank);
      }, 
      tuple<>()
    );
  }

  // f(int dest_seg, int index, int source_seg, int rank, tuple<...> desc)
  // returns the number of work-items to create.
  template<typename func_t, typename pointers_t, typename... args_t>
  mem_t<int> downsweep(func_t f, pointers_t caching_iterators, args_t... args) {
    // Input
    const int* mp_data = mp.data();
    const short* bits_data = bits.data();
    const int2* counts_data = cta_offsets.data();
    int count = this->count;
    auto segments = this->segments;
    int num_segments = this->num_segments;

    // Output.
    int num_dest_segments = cta_total.x;
    mem_t<int> segments_result(num_dest_segments, context);
    int* segments_output = segments_result.data();

   // typedef tuple_iterator_value_t<pointers_t> value_t;
   // typedef tuple<int> value_t;
    auto downsweep_k = [=]MGPU_DEVICE(int tid, int cta, args_t... args) {
      typedef typename launch_t::sm_ptx params_t;
      enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };
      typedef cta_scan_t<nt, int> scan_t;
      
      // Note that this is a struct rather than the typical union. We want
      // all three kinds of things to be valid during the callbacks into
      // f.
      __shared__ struct {
        int indices[nv + 2];
        short targets[nv];
        typename scan_t::storage_t scan;
      } shared;

      // Decode the bits signifying work creation and compact them.
      int segment_bits = bits_data[nt * cta + tid];
      strided_iterate<nt, vt>([&](int i, int j) {
        int work_create = 0 != ((1<< i) & segment_bits);
        shared.indices[j] = work_create;
      }, tid);
      __syncthreads();

      // Do a parallel scan of the work-create flags. Compact the indices
      // of the work-creating items into shared.targets.
      array_t<int, vt> flags = shared_to_reg_thread<nt, vt>(
        shared.indices, tid);
      scan_result_t<int> scan = scan_t().scan(tid, reduce(flags), shared.scan);
      iterate<vt>([&](int i) {
        if(flags[i]) shared.targets[scan.scan++] = (short)(vt * tid + i);
      });
      
      // Use load-balancing search to fill shared memory with the segment of
      // each in-range work-item.
      lbs_fill_t fill = cta_load_balance_fill<nt, vt>(count, segments,
        num_segments, tid, cta, mp_data, shared.indices);
      const int* a_shared = shared.indices;
      const int* b_shared = shared.indices + fill.b_offset;

      int num_items = scan.reduction;
      int segments_dest = counts_data[cta].x;
      int work_item_dest = counts_data[cta].y;

      int num_rounds = div_up(num_items, nt);
      for(int i = 0; i < num_rounds; ++i) {
        int j = i * nt + tid;
        int dest_seg = segments_dest + j;
        int work_count = 0;
        if(j < num_items) {
          // Lookup the segment info.
          int cta_index = shared.targets[j];
          int seg = a_shared[cta_index];
          int seg_begin = b_shared[seg];
          int index = fill.range.a_begin + cta_index;
          int rank = index - seg_begin;

          // Invoke the callback and the get the work-item count.
          tuple<int> cached = load(caching_iterators, seg);
          work_count = f(dest_seg, index, seg, rank, cached, args...);
        }

        // Scan the work-counts.
        scan_result_t<int> work_scan = scan_t().scan(tid, work_count,
          shared.scan);

        // Stream the segments-descriptor array.
        if(j < num_items)
          segments_output[dest_seg] = work_item_dest + work_scan.scan;
        work_item_dest += work_scan.reduction;
      }
    };
    cta_launch<launch_t>(downsweep_k, num_ctas, context, args...);

    return segments_result;     
  }

  template<typename func_t, typename... args_t>
  mem_t<int> downsweep(func_t f, args_t... args) {
    return downsweep(
      [=]MGPU_DEVICE(int dest_seg, int index, int seg, int rank, tuple<>,
        args_t... args) {
        return f(dest_seg, index, seg, rank, args...);
      },
      tuple<>(), args...
    );
  }
};

// Use lbs_workcreate to construct an expt::workcreate_t instance. Then call
// upsweep and downsweep, providing an appropriate lambda function.
template<typename launch_arg_t = empty_t, typename segments_it>
workcreate_t<launch_arg_t, segments_it>
lbs_workcreate(int count, segments_it segments, int num_segments,
  context_t& context) {
  return workcreate_t<launch_arg_t, segments_it> {
    count, segments, num_segments, context
  };
}

} // namespace expt

END_MGPU_NAMESPACE
