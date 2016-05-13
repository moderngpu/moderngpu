// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once
#include "cta_load_balance.hxx"
#include "search.hxx"

BEGIN_MGPU_NAMESPACE

template<typename launch_arg_t = empty_t, typename func_t, 
  typename segments_it, typename pointers_t, typename... args_t>
void transform_lbs(func_t f, int count, segments_it segments, 
  int num_segments, pointers_t caching_iterators, context_t& context,
  args_t... args) {

  typedef typename conditional_typedef_t<launch_arg_t, 
    launch_box_t<
      arch_20_cta<128, 11, 9>,
      arch_35_cta<128,  7, 5>,
      arch_52_cta<128, 11, 9>
    >
  >::type_t launch_t;

  typedef typename std::iterator_traits<segments_it>::value_type int_t;
  typedef tuple_iterator_value_t<pointers_t> value_t;

  mem_t<int_t> mp = load_balance_partitions(count, segments, num_segments,
    launch_t::nv(context), context);
  const int_t* mp_data = mp.data();

  auto k = [=]MGPU_DEVICE(int tid, int cta, args_t... args) {

    typedef typename launch_t::sm_ptx params_t;
    enum { nt = params_t::nt, vt = params_t::vt, vt0 = params_t::vt0 };
    typedef cta_load_balance_t<nt, vt> load_balance_t;
    typedef detail::cached_segment_load_t<nt, pointers_t> cached_load_t;

    __shared__ union {
      typename load_balance_t::storage_t lbs;
      typename cached_load_t::storage_t cached;
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
    strided_iterate<nt, vt, vt0>([=](int i, int j) {
      int index = lbs.merge_range.a_begin + j;
      int seg = lbs.segments[i];
      int rank = lbs.ranks[i];

      f(index, seg, rank, cached_values[i], args...);
    }, tid, lbs.merge_range.a_count());
  };
  cta_transform<launch_t>(k, count + num_segments, context, args...);
}

// load-balancing search without caching.
template<typename launch_arg_t = empty_t, typename func_t, 
  typename segments_it, typename... args_t>
void transform_lbs(func_t f, int count, segments_it segments, 
  int num_segments, context_t& context, args_t... args) {

  transform_lbs<launch_arg_t>(
    [=]MGPU_DEVICE(int index, int seg, int rank, tuple<>, args_t... args) {
      f(index, seg, rank, args...);    // drop the cached values.
    },
    count, segments, num_segments, tuple<>(), context, args...
  );
}

template<typename launch_arg_t = empty_t, typename segments_it,
  typename output_it>
void load_balance_search(int count, segments_it segments, 
  int num_segments, output_it output, context_t& context) {

  transform_lbs<launch_arg_t>([=]MGPU_DEVICE(int index, int seg, int rank) {
    output[index] = seg;
  }, count, segments, num_segments, context);
}

END_MGPU_NAMESPACE
