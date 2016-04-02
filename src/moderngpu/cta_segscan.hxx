// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once

#include "cta_scan.hxx"

BEGIN_MGPU_NAMESPACE

template<typename type_t>
struct segscan_result_t {
  type_t scan;
  type_t reduction;
  bool has_carry_in;
  int left_lane;
};

template<int nt, typename type_t>
struct cta_segscan_t {
  enum { num_warps = nt / warp_size };

  union storage_t {
    int delta[num_warps + nt]; 
    struct { type_t values[2 * nt]; int packed[nt]; };
  };

  MGPU_DEVICE int find_left_lane(int tid, bool has_head_flag, 
    storage_t& storage) const {

    int warp = tid / warp_size;
    int lane = (warp_size - 1) & tid;
    int warp_mask = 0xffffffff>> (31 - lane);   // inclusive search.
    int cta_mask = 0x7fffffff>> (31 - lane);    // exclusive search.

    // Build a head flag bitfield and store it into shared memory.
    int warp_bits = __ballot(has_head_flag);
    storage.delta[warp] = warp_bits;
    __syncthreads();

    if(tid < num_warps) {
      int cta_bits = __ballot(0 != storage.delta[tid]);
      int warp_segment = 31 - clz(cta_mask & cta_bits);
      int start = (-1 != warp_segment) ?
        (31 - clz(storage.delta[warp_segment]) + 32 * warp_segment) : 0;
      storage.delta[num_warps + tid] = start;
    }
    __syncthreads();

    // Find the closest flag to the left of this thread within the warp.
    // Include the flag for this thread.
    int start = 31 - clz(warp_mask & warp_bits);
    if(-1 != start) start += ~31 & tid;
    else start = storage.delta[num_warps + warp];
    __syncthreads();

    return start;
  }

  template<typename op_t = plus_t<type_t> >
  MGPU_DEVICE segscan_result_t<type_t> segscan(int tid, bool has_head_flag,
    bool has_carry_out, type_t x, storage_t& storage, type_t init = type_t(),
    op_t op = op_t()) const {

    if(!has_carry_out) x = init;

    int left_lane = find_left_lane(tid, has_head_flag, storage);
    int tid_delta = tid - left_lane;

    // Store the has_carry_out flag.
    storage.packed[tid] = (int)has_carry_out | (left_lane<< 1);

    // Run an inclusive scan.
    int first = 0;
    storage.values[first + tid] = x;
    __syncthreads();

    int packed = storage.packed[left_lane];
    left_lane = packed>> 1;
    tid_delta = tid - left_lane;
    if(0 == (1 & packed)) --tid_delta;

    iterate<s_log2(nt)>([&](int pass) {
      int offset = 1<< pass;
      if(tid_delta >= offset)
        x = op(x, storage.values[first + tid - offset]);
      first = nt - first;
      storage.values[first + tid] = x;
      __syncthreads();
    });

    // Get the exclusive scan by fetching the preceding element. Also return
    // the carry-out value as the total.
    bool has_carry_in = tid ? (0 != (1 & storage.packed[tid - 1])) : false;

    segscan_result_t<type_t> result { 
      (has_carry_in && tid) ? storage.values[first + tid - 1] : init,
      storage.values[first + nt - 1],
      has_carry_in,
      left_lane
    };
    __syncthreads();

    return result;
  }
};

END_MGPU_NAMESPACE
