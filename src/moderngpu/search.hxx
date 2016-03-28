// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once

#include "loadstore.hxx"
#include "operators.hxx"
#include "cta_search.hxx"
#include "memory.hxx"
#include "context.hxx"

BEGIN_MGPU_NAMESPACE

template<bounds_t bounds, typename a_keys_it, typename b_keys_it,
  typename comp_t>
mem_t<int> merge_path_partitions(a_keys_it a, int64_t a_count, b_keys_it b,
  int64_t b_count, int64_t spacing, comp_t comp, context_t& context) {

  typedef int int_t;
  int num_partitions = (int)div_up(a_count + b_count, spacing) + 1;
  mem_t<int_t> mem(num_partitions, context);
  int_t* p = mem.data();
  transform([=]MGPU_DEVICE(int index) {
    int_t diag = (int_t)min(spacing * index, a_count + b_count);
    p[index] = merge_path<bounds>(a, (int_t)a_count, b, (int_t)b_count,
      diag, comp);
  }, num_partitions, context);
  return mem;
}

template<typename segments_it>
auto load_balance_partitions(int64_t dest_count, segments_it segments, 
  int num_segments, int spacing, context_t& context) -> 
  mem_t<typename std::iterator_traits<segments_it>::value_type> {

  typedef typename std::iterator_traits<segments_it>::value_type int_t;
  return merge_path_partitions<bounds_upper>(counting_iterator_t<int_t>(0), 
    dest_count, segments, num_segments, spacing, less_t<int_t>(), context);
}

template<bounds_t bounds, typename keys_it>
mem_t<int> binary_search_partitions(keys_it keys, int count, int num_items,
  int spacing, context_t& context) {

  int num_partitions = div_up(count, spacing) + 1;
  mem_t<int> mem(num_partitions, context);
  int* p = mem.data();
  transform([=]MGPU_DEVICE(int index) {
    int key = min(spacing * index, count);
    p[index] = binary_search<bounds>(keys, num_items, key, less_t<int>());
  }, num_partitions, context);
  return mem;
}

END_MGPU_NAMESPACE
