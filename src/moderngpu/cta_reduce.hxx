// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once
#include "loadstore.hxx"
#include "intrinsics.hxx"

BEGIN_MGPU_NAMESPACE

// requires __CUDA_ARCH__ >= 300.
// warp_size can be any power-of-two <= warp_size.
// warp_reduce_t returns the reduction only in lane 0.
template<typename type_t, int group_size>
struct shfl_reduce_t {
 
  static_assert(group_size <= warp_size && is_pow2(group_size),
    "shfl_reduce_t must operate on a pow2 number of threads <= warp_size (32)");
  enum { num_passes = s_log2(group_size) };

  template<typename op_t = plus_t<type_t> >
  MGPU_DEVICE type_t reduce(int lane, type_t x, int count, op_t op = op_t()) {
    if(count == group_size) { 
      iterate<num_passes>([&](int pass) {
        int offset = 1<< pass;
        x = shfl_down_op(x, offset, op, group_size);
      });
    } else {
      iterate<num_passes>([&](int pass) {
        int offset = 1<< pass;
        type_t y = shfl_down(x, offset, group_size);
        if(lane + offset < count) x = op(x, y);
      });
    }
    return x;
  }
};

// cta_reduce_t returns the reduction of all inputs for thread 0, and returns
// type_t() for all other threads. This behavior saves a broadcast.

template<int nt, typename type_t>
struct cta_reduce_t {

  enum { 
    group_size = min(nt, (int)warp_size), 
    num_passes = s_log2(group_size),
    num_items = nt / group_size 
  };

  static_assert(0 == nt % warp_size, 
    "cta_reduce_t requires num threads to be a multiple of warp_size (32)");

  struct storage_t {
    struct { type_t data[max(nt, 2 * group_size)]; };
  };

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300

  typedef shfl_reduce_t<type_t, group_size> group_reduce_t;

  template<typename op_t = plus_t<type_t> >
  MGPU_DEVICE type_t reduce(int tid, type_t x, storage_t& storage, 
    int count = nt, op_t op = op_t(), bool all_return = true) const {

    // Store your data into shared memory.
    storage.data[tid] = x;
    __syncthreads();

    if(tid < group_size) {
      // Each thread scans within its lane.
      strided_iterate<group_size, num_items>([&](int i, int j) {
        if(i > 0) x = op(x, storage.data[j]);
      }, tid, count);

      // Cooperative reduction.
      x = group_reduce_t().reduce(tid, x, min(count, (int)group_size), op);

      if(all_return) storage.data[tid] = x;
    }
    __syncthreads();

    if(all_return) {
      x = storage.data[0];
      __syncthreads();
    }
    return x;
  }

#else

  template<typename op_t = plus_t<type_t> >
  MGPU_DEVICE type_t reduce(int tid, type_t x, storage_t& storage, 
    int count = nt, op_t op = op_t(), bool all_return = true) const {

    // Store your data into shared memory.
    storage.data[tid] = x;
    __syncthreads();

    if(tid < group_size) {
      // Each thread scans within its lane.
      strided_iterate<group_size, num_items>([&](int i, int j) {
        type_t y = storage.data[j];
        if(i > 0) x = op(x, y);
      }, tid, count);
      storage.data[tid] = x;
    }
    __syncthreads();

    int count2 = min(count, int(group_size));
    int first = (1 & num_passes) ? group_size : 0;
    if(tid < group_size)
      storage.data[first + tid] = x;
    __syncthreads();

    iterate<num_passes>([&](int pass) {
      if(tid < group_size) {
        int offset = 1 << pass;
        if(tid + offset < count2) 
          x = op(x, storage.data[first + offset + tid]);
        first = group_size - first;
        storage.data[first + tid] = x;
      }
      __syncthreads();
    });

    if(all_return) {
      x = storage.data[0];
      __syncthreads();
    }
    return x;
  }

#endif
};

END_MGPU_NAMESPACE
