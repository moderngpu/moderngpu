// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once
#include "operators.hxx"

BEGIN_MGPU_NAMESPACE

////////////////////////////////////////////////////////////////////////////////
// Odd-even transposition sorting network. Sorts keys and values in-place in
// register.
// http://en.wikipedia.org/wiki/Odd%E2%80%93even_sort

template<typename type_t, int vt, typename comp_t>
MGPU_HOST_DEVICE array_t<type_t, vt> 
odd_even_sort(array_t<type_t, vt> x, comp_t comp, int flags = 0) { 
  iterate<vt>([&](int I) {
    PRAGMA_UNROLL
    for(int i = 1 & I; i < vt - 1; i += 2) {
      if((0 == ((2<< i) & flags)) && comp(x[i + 1], x[i]))
        swap(x[i], x[i + 1]);
    }
  });
  return x;
}

template<typename key_t, typename val_t, int vt, typename comp_t>
MGPU_HOST_DEVICE kv_array_t<key_t, val_t, vt> 
odd_even_sort(kv_array_t<key_t, val_t, vt> x, comp_t comp, int flags = 0) { 
  iterate<vt>([&](int I) {
    PRAGMA_UNROLL
    for(int i = 1 & I; i < vt - 1; i += 2) {
      if((0 == ((2<< i) & flags)) && comp(x.keys[i + 1], x.keys[i])) {
        swap(x.keys[i], x.keys[i + 1]);
        swap(x.vals[i], x.vals[i + 1]);
      }
    }
  });
  return x;
}

////////////////////////////////////////////////////////////////////////////////
// TODO: Batcher Odd-Even Mergesort network
// Unstable but executes much faster than the transposition sort.
// http://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort
#if 0
template<int width, int low, int count>
struct odd_even_mergesort_t {

};

template<typename key_t, typename val_t, int vt, typename comp_t>
MGPU_HOST_DEVICE kv_array_t<key_t, val_t, vt> 
odd_even_mergesort(kv_array_t<key_t, val_t, vt> x, int flags = 0) {
  return kv_array_t<key_t, val_t, vt>();
}
#endif

END_MGPU_NAMESPACE
