// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once
#include "search.hxx"
#include "cta_load_balance.hxx"
#include "cta_segscan.hxx"
#include "transform.hxx"
#include "memory.hxx"

BEGIN_MGPU_NAMESPACE

namespace detail {

////////////////////////////////////////////////////////////////////////////////
// cta_segreduce_t is common intra-warp segmented reduction code for 
// these kernels. Should clean up and move to cta_segreduce.hxx.

template<int nt, int vt, typename type_t>
struct cta_segreduce_t {
  typedef cta_segscan_t<nt, type_t> segscan_t;
  
  union storage_t {
    typename segscan_t::storage_t segscan;
    type_t values[nt * vt + 1];
  };

  // Values must be stored in storage.values on entry.
  template<typename op_t, typename output_it>
  MGPU_DEVICE void segreduce(merge_range_t merge_range, 
    lbs_placement_t placement, array_t<bool, vt + 1> p, int tid, 
    int cta, type_t init, op_t op, output_it output, 
    type_t* carry_out_values, int* carry_out_codes, storage_t& storage) {

    int cur_item = placement.a_index;
    int begin_segment = placement.b_index;
    int cur_segment = begin_segment;
    bool carry_in = false;

    const type_t* a_shared = storage.values - merge_range.a_begin;
    type_t x[vt];
    int segments[vt + 1];
    iterate<vt>([&](int i) {
      if(p[i]) {
        // This is a data node, so accumulate and advance the data ID.
        x[i] = a_shared[cur_item++];
        if(carry_in) x[i] = op(x[i - 1], x[i]);
        carry_in = true;
      } else {
        // This is a segment node, so advance the segment ID.
        x[i] = init;
        ++cur_segment;
        carry_in = false;
      }
      segments[i] = cur_segment;
    });
    // Always flush at the end of the last thread.
    bool overwrite = (nt - 1 == tid) && (!p[vt - 1] && p[vt]);
    if(nt - 1 == tid) p[vt] = false;
    if(!p[vt]) ++cur_segment;
    segments[vt] = cur_segment;
    overwrite = __syncthreads_or(overwrite);

    // Get the segment ID for the next item. This lets us find an end flag
    // for the last value in this thread.
    bool has_head_flag = begin_segment < segments[vt - 1];
    bool has_carry_out = p[vt - 1];

    // Compute the carry-in for each thread.
    segscan_result_t<type_t> result = segscan_t().segscan(tid, has_head_flag,
      has_carry_out, x[vt - 1], storage.segscan, init, op);

    // Add the carry-in back into each value and recompute the reductions.
    type_t* x_shared = storage.values - placement.range.b_begin;
    carry_in = result.has_carry_in && p[0];
    iterate<vt>([&](int i) {
      if(segments[i] < segments[i + 1]) {
        // We've hit the end of this segment. Store the reduction to shared
        // memory.
        if(carry_in) x[i] = op(result.scan, x[i]);
        x_shared[segments[i]] = x[i];
        carry_in = false;
      }
    });
    __syncthreads();

    // Store the reductions for segments which begin in this tile. 
    for(int i = merge_range.b_begin + tid; i < merge_range.b_end; i += nt)
      output[i] = x_shared[i];

    // Store the partial reduction for the segment which begins in the 
    // preceding tile, if there is one.
    if(!tid) {
      if(segments[0] == merge_range.b_begin) segments[0] = -1;
      int code = (segments[0]<< 1) | (int)overwrite;
      carry_out_values[cta] = (segments[0] != -1) ?
        x_shared[segments[0]] : 
        init;
      carry_out_codes[cta] = code;
    }
  }
};

////////////////////////////////////////////////////////////////////////////////
// Adds the carry-out for each segreduce CTA into the outputs.

template<typename output_it, typename type_t, typename op_t>
void segreduce_fixup(output_it output, const type_t* values,
  const int* codes, int count, op_t op, type_t init,
  context_t& context) {

  enum { nt = 512 };
  int num_ctas = div_up(count, nt);

  mem_t<type_t> carry_out(num_ctas, context);
  mem_t<int> codes_out(num_ctas, context);
  type_t* carry_out_data = carry_out.data();
  int* codes_data = codes_out.data();

  auto k_fixup = [=]MGPU_DEVICE(int tid, int cta) {
    typedef cta_segscan_t<nt, type_t> segscan_t;
    __shared__ struct {
      bool head_flags[nt];
      typename segscan_t::storage_t segscan;
    } shared;

    range_t tile = get_tile(cta, nt, count);
    int gid = tile.begin + tid;

    ////////////////////////////////////////////////////////////////////////////
    // As in the outer segmented reduce kernel, update the reductions for all
    // segments that *start* in this CTA. That is, the first carry-out code
    // for a segment must be mapped into this CTA to actually apply the 
    // accumulate. This CTA will return a partial reduction for the segment
    // that overlaps this CTA but starts in a preceding CTA.

    // We don't need to worry about storing new overwrite bits as this kernel
    // will always add carry-in values to empty segments.

    int code0 = (gid - 1 >= 0 && gid - 1 < count) ? codes[gid - 1] : -1;
    int code1 = (gid < count) ? codes[gid] : -1;
    int code2 = (gid + 1 < count) ? codes[gid + 1] : -1;
    type_t value = (gid < count) ? values[gid] : init;

    int seg0 = code0>> 1;
    int seg1 = code1>> 1;
    int seg2 = code2>> 1;
    bool has_head_flag = seg0 != seg1 || -1 == seg1;
    bool has_carry_out = -1 != seg1 && seg1 == seg2;
    bool has_end_flag = seg1 != seg2;

    // Put the head flag in shared memory, because the last thread 
    // participating in a reduction in the CTA needs to check the head flag
    // for the first thread in the reduction.
    shared.head_flags[tid] = has_head_flag;

    segscan_result_t<type_t> result = segscan_t().segscan(tid, has_head_flag,
      has_carry_out, value, shared.segscan, init, op);

    bool carry_out_written = false;
    if(-1 != seg1 && (has_end_flag || nt - 1 == tid)) {
      // This is a valid reduction.
      if(result.has_carry_in) 
        value = op(value, result.scan);

      if(0 == result.left_lane && !shared.head_flags[result.left_lane]) {
        carry_out_data[cta] = value;
        codes_data[cta] = seg1<< 1;
        carry_out_written = true;
      } else {
        int left_code = codes[tile.begin + result.left_lane - 1];
        if(0 == (1 & left_code))     // Add in the value already stored.
          value = op(value, output[seg1]);
        output[seg1] = value;
      }
    }

    carry_out_written = __syncthreads_or(carry_out_written);
    if(!carry_out_written && !tid)
      codes_data[cta] = -1<< 1;
  };
  cta_launch<nt>(k_fixup, num_ctas, context);

  if(num_ctas > 1)
    segreduce_fixup(output, carry_out_data, codes_data, 
      num_ctas, op, init, context);
}

} // namespace detail

////////////////////////////////////////////////////////////////////////////////
// Segmented reduction with loading from an input iterator. This does not
// require explicit materialization of the load-balancing search.

template<typename launch_arg_t = empty_t, typename input_it,
  typename segments_it, typename output_it, typename op_t, typename type_t>
void segreduce(input_it input, int count, segments_it segments, 
  int num_segments, output_it output, op_t op, type_t init, 
  context_t& context) {

  typedef typename conditional_typedef_t<launch_arg_t, 
    launch_box_t<
      arch_20_cta<128, 11, 8>,
      arch_35_cta<128,  7, 5>,
      arch_52_cta<128, 11, 8>
    >
  >::type_t launch_t;

  cta_dim_t cta_dim = launch_t::cta_dim(context);
  int num_ctas = cta_dim.num_ctas(count + num_segments);

  mem_t<type_t> carry_out(num_ctas, context);
  mem_t<int> codes(num_ctas, context);
  type_t* carry_out_data = carry_out.data();
  int* codes_data = codes.data();

  mem_t<int> mp = load_balance_partitions(count, segments, num_segments,
    cta_dim.nv(), context);
  const int* mp_data = mp.data();

  auto k_reduce = [=]MGPU_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum { nt = params_t::nt, vt = params_t::vt, vt0 = params_t::vt0 };
    typedef detail::cta_segreduce_t<nt, vt, type_t> segreduce_t;

    __shared__ union {
      typename segreduce_t::storage_t segreduce;
      type_t values[nt * vt + 1];
      type_t indices[nt * vt + 2];
    } shared;

    merge_range_t merge_range = compute_merge_range(count, num_segments, 
      cta, nt * vt, mp_data[cta], mp_data[cta + 1]);

    // Cooperatively load values from input into shared.
    mem_to_shared<nt, vt, vt0>(input + merge_range.a_begin, tid, 
      merge_range.a_count(), shared.segreduce.values);

    // Load segment data into the B region of shared. Search for the starting
    // index of each thread for a merge.
    int* b_shared = sizeof(type_t) > sizeof(int) ?
      (int*)(shared.segreduce.values + merge_range.a_count()) :
      ((int*)shared.segreduce.values + merge_range.a_count());
    lbs_placement_t placement = cta_load_balance_place<nt, vt>(tid, 
      merge_range, count, segments, num_segments, b_shared);

    // Adjust the pointer so that dereferencing at the segment ID returns the
    // offset of that segment.
    b_shared -= placement.range.b_begin;
    int cur_item = placement.a_index;
    int cur_segment = placement.b_index;
    array_t<bool, vt + 1> merge_bits;
    iterate<vt + 1>([&](int i) {
      bool p = cur_item < b_shared[cur_segment + 1];
      if(p) ++cur_item;
      else ++cur_segment;
      merge_bits[i] = p;
    });

    // Compute the segmented reduction.
    segreduce_t().segreduce(merge_range, placement, merge_bits, tid, cta, 
      init, op, output, carry_out_data, codes_data, shared.segreduce);

  };
  cta_launch<launch_t>(k_reduce, num_ctas, context);

  if(num_ctas > 1)
    detail::segreduce_fixup(output, carry_out_data, codes_data, num_ctas,
      op, init, context);
}

////////////////////////////////////////////////////////////////////////////////

template<typename launch_arg_t = empty_t, typename func_t,
  typename segments_it, typename output_it, typename op_t, typename type_t>
void transform_segreduce(func_t f, int count, segments_it segments, 
  int num_segments, output_it output, op_t op, type_t init, 
  context_t& context) {

  segreduce<launch_arg_t>(make_load_iterator<type_t>(f), count, segments, 
    num_segments, output, op, init, context);
}

////////////////////////////////////////////////////////////////////////////////
// spmv - sparse matrix * vector.

template<typename launch_arg_t = empty_t, typename matrix_it,
  typename columns_it, typename vector_it, typename segments_it, 
  typename output_it>
void spmv(matrix_it matrix, columns_it columns, vector_it vector,
  int count, segments_it segments, int num_segments, output_it output,
  context_t& context) { 

  typedef typename std::iterator_traits<matrix_it>::value_type type_t;
  
  transform_segreduce<launch_arg_t>([=]MGPU_DEVICE(int index) {
    return matrix[index] * ldg(vector + columns[index]);    // sparse m * v.
  }, count, segments, num_segments, output, plus_t<type_t>(), 
    (type_t)0, context);
}

////////////////////////////////////////////////////////////////////////////////
// lbs_segreduce

template<typename launch_arg_t = empty_t, 
  typename func_t, typename segments_it, typename pointers_t, 
  typename output_it, typename op_t, typename type_t, typename... args_t>
void lbs_segreduce(func_t f, int count, segments_it segments,
  int num_segments, pointers_t caching_iterators, output_it output, op_t op,
  type_t init, context_t& context, args_t... args) {

  typedef typename conditional_typedef_t<launch_arg_t, 
    launch_box_t<
      arch_20_cta<128, 11, 8>,
      arch_35_cta<128,  7, 5>,
      arch_52_cta<128, 11, 8>
    >
  >::type_t launch_t;

  typedef tuple_iterator_value_t<pointers_t> value_t;

  cta_dim_t cta_dim = launch_t::cta_dim(context);
  int num_ctas = cta_dim.num_ctas(count + num_segments);

  mem_t<type_t> carry_out(num_ctas, context);
  mem_t<int> codes(num_ctas, context);
  type_t* carry_out_data = carry_out.data();
  int* codes_data = codes.data();

  mem_t<int> mp = load_balance_partitions(count, segments, num_segments,
    cta_dim.nv(), context);
  const int* mp_data = mp.data();

  auto k_reduce = [=]MGPU_DEVICE(int tid, int cta, args_t... args) {
    typedef typename launch_t::sm_ptx params_t;
    enum { nt = params_t::nt, vt = params_t::vt, vt0 = params_t::vt0 };
    typedef cta_load_balance_t<nt, vt> load_balance_t;
    typedef detail::cached_segment_load_t<nt, pointers_t> cached_load_t;
    typedef detail::cta_segreduce_t<nt, vt, type_t> segreduce_t;

    __shared__ union {
      typename load_balance_t::storage_t lbs;
      typename cached_load_t::storage_t cached;
      typename segreduce_t::storage_t segreduce;
      type_t values[nt * vt + 1];
    } shared;

    // Compute the load-balancing search and materialize (index, seg, rank)
    // arrays.
    auto lbs = load_balance_t().load_balance(count, segments, num_segments,
      tid, cta, mp_data, shared.lbs);

    // Load from the cached iterators. Use the placement range, not the 
    // merge-path range for situating the segments.
    array_t<value_t, vt> cached_values = cached_load_t::template load<vt0>(
      tid, lbs.merge_range.a_count(), lbs.placement.range.b_range(), 
      lbs.segments, shared.cached, caching_iterators);

    // Call the user-supplied functor f.
    array_t<type_t, vt> strided_values;
    strided_iterate<nt, vt, vt0>([&](int i, int j) {
      int index = lbs.merge_range.a_begin + j;
      int seg = lbs.segments[i];
      int rank = lbs.ranks[i];

      strided_values[i] = f(index, seg, rank, cached_values[i], args...);
    }, tid, lbs.merge_range.a_count());

    // Store the values back to shared memory for segmented reduction.
    reg_to_shared_strided<nt, vt>(strided_values, tid, 
      shared.segreduce.values);

    // Split the flags.
    array_t<bool, vt + 1> merge_bits;
    iterate<vt + 1>([&](int i) {
      merge_bits[i] = 0 != ((1<< i) & lbs.merge_flags);
    });

    // Compute the segmented reduction.
    segreduce_t().segreduce(lbs.merge_range, lbs.placement, merge_bits,
      tid, cta, init, op, output, carry_out_data, codes_data, 
      shared.segreduce);
  };
  cta_launch<launch_t>(k_reduce, num_ctas, context, args...);

  if(num_ctas > 1)
    detail::segreduce_fixup(output, carry_out_data, codes_data, num_ctas,
      op, init, context);
}

// lbs_segreduce with no caching iterators.
template<typename launch_arg_t = empty_t, 
  typename func_t, typename segments_it, typename output_it, typename op_t,
  typename type_t, typename... args_t>
void lbs_segreduce(func_t f, int count, segments_it segments,
  int num_segments, output_it output, op_t op, type_t init, 
  context_t& context, args_t... args) {

  lbs_segreduce<launch_arg_t>(
    [=]MGPU_DEVICE(int index, int seg, int rank, tuple<>, args_t... args) {
      return f(index, seg, rank, args...);
    },
    count, segments, num_segments, tuple<>(), output, op, init, context,
    args...
  );
}

END_MGPU_NAMESPACE
