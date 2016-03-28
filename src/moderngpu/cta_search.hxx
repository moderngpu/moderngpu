// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once

#include "cta_merge.hxx"

BEGIN_MGPU_NAMESPACE

template<bounds_t bounds, typename keys_it, typename int_t, typename key_t, 
  typename comp_t>
MGPU_HOST_DEVICE int_t binary_search(keys_it keys, int_t count, key_t key,
  comp_t comp) {

  int_t begin = 0;
  int_t end = count;
  while(begin < end) {
    int_t mid = (begin + end) / 2;
    key_t key2 = keys[mid];
    bool pred = (bounds_upper == bounds) ? 
      !comp(key, key2) :
      comp(key2, key);
    if(pred) begin = mid + 1;
    else end = mid;
  }
  return begin;
}

////////////////////////////////////////////////////////////////////////////////
// TODO: Implement a moderngpu V1 style vectorized sorted search.

template<typename type_t, int vt>
struct search_result_t {
  array_t<type_t, vt> keys;
  array_t<int, vt> indices;
  int decisions;              // Set a bit if this iteration has progressed A.
  int matches_a;              // A set flag for a match on each iteration.
  int matches_b;
};

template<int vt, bounds_t bounds, bool range_check, typename type_t, 
  typename comp_t>
MGPU_DEVICE search_result_t<type_t, vt> 
serial_search(const type_t* keys_shared, merge_range_t range,
  int a_offset, int b_offset, comp_t comp, bool sync = true) {

  type_t a_key = keys_shared[range.a_begin];
  type_t b_key = keys_shared[range.b_begin];
  type_t a_prev = type_t(), b_prev = type_t();

  int a_start = 0;
  int b_start = range.a_end;    // Assume the b_keys start right after the end
                                // of the a_keys.
  if(range.a_begin > 0) a_prev = keys_shared[range.a_begin - 1];
  if(range.b_begin > b_start) b_prev = keys_shared[range.b_begin - 1];

  search_result_t<type_t, vt> result = search_result_t<type_t, vt>();

  iterate<vt>([&](int i) {
    // This is almost the same body as serial_merge, except for the match
    // criterion below.
    bool p = merge_predicate<bounds, range_check>(a_key, b_key, range, comp);

    if(p) {
      bool match = (bounds_upper == bounds) ?
        (!range_check || range.b_begin > b_start) && 
          !comp(b_prev, a_key) :
        (!range_check || range.b_valid()) && 
          !comp(a_key, b_key);

      result.decisions |= 1<< i;
      result.matches_a |= (int)match<< i;
      a_prev = a_key;

    } else {
      bool match = (bounds_upper == bounds) ?
        (!range_check || (range.a_valid() && range.b_valid())) && 
          !comp(b_key, a_key) :
        (!range_check || (range.b_valid() && range.a_begin > a_start)) && 
          !comp(a_prev, b_key);

      result.matches_b |= (int)match<< i;
      b_prev = b_key;
    }

    // Same advancement behavior as serial_merge.
    int index = p ? range.a_begin : range.b_begin;

    result.keys[i] = p ? a_key : b_key;
    result.indices[i] = index + (p ? a_offset : b_offset);

    type_t c_key = keys_shared[++index];
    if(p) a_key = c_key, range.a_begin = index;
    else b_key = c_key, range.b_begin = index;
  });

  if(sync) __syncthreads();

  return result;
}

END_MGPU_NAMESPACE
