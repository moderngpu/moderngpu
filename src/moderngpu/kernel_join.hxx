// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once
#include "kernel_sortedsearch.hxx"
#include "kernel_scan.hxx"
#include "kernel_load_balance.hxx"

BEGIN_MGPU_NAMESPACE

template<typename launch_arg_t = empty_t, 
  typename a_it, typename b_it, typename comp_t>
mem_t<int2> inner_join(a_it a, int a_count, b_it b, int b_count, 
  comp_t comp, context_t& context) {

  // Compute lower and upper bounds of a into b.
  mem_t<int> lower(a_count, context);
  mem_t<int> upper(a_count, context);
  sorted_search<bounds_lower, launch_arg_t>(a, a_count, b, b_count, 
    lower.data(), comp, context);
  sorted_search<bounds_upper, launch_arg_t>(a, a_count, b, b_count, 
    upper.data(), comp, context);

  // Compute output ranges by scanning upper - lower. Retrieve the reduction
  // of the scan, which specifies the size of the output array to allocate.
  mem_t<int> scanned_sizes(a_count, context);
  const int* lower_data = lower.data();
  const int* upper_data = upper.data();

  mem_t<int> count(1, context);
  transform_scan<int>([=]MGPU_DEVICE(int index) {
    return upper_data[index] - lower_data[index];
  }, a_count, scanned_sizes.data(), plus_t<int>(), count.data(), context);

  // Allocate an int2 output array and use load-balancing search to compute
  // the join.
  int join_count = from_mem(count)[0];
  mem_t<int2> output(join_count, context);
  int2* output_data = output.data();

  // Use load-balancing search on the segmens. The output is a pair with
  // a_index = seg and b_index = lower_data[seg] + rank.
  auto k = [=]MGPU_DEVICE(int index, int seg, int rank, tuple<int> lower) {
    output_data[index] = make_int2(seg, get<0>(lower) + rank);
  };
  transform_lbs<launch_arg_t>(k, join_count, scanned_sizes.data(), a_count,
    make_tuple(lower_data), context);

  return output;
}

END_MGPU_NAMESPACE
