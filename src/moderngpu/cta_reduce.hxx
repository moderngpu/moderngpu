// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once
#include "loadstore.hxx"
#include "intrinsics.hxx"

BEGIN_MGPU_NAMESPACE

// cta_reduce_t returns the reduction of all inputs for thread 0, and returns
// type_t() for all other threads. This behavior saves a broadcast.

template<int nt, typename type_t, typename op_t = plus_t<type_t>,
  bool optimized = 
    std::is_same<plus_t<int>, op_t>::value || 
    std::is_same<plus_t<float>, op_t>::value
>
struct cta_reduce_t {
  struct storage_t {
    type_t data[nt];
  };
	
  MGPU_DEVICE type_t reduce(int tid, type_t x, storage_t& storage, 
    int count = nt, op_t op = op_t()) const {

    storage.data[tid] = x;
    __syncthreads();

    // Fold the data in half with each pass.
    iterate<s_log2(nt)>([&](int pass) {
      int dest_count = nt>> (pass + 1);
      if((tid < dest_count) && (dest_count + tid < count)) {
        // Read from the right half and store to the left half.
        x = op(x, storage.data[dest_count + tid]);
        storage.data[tid] = x;
      }
      __syncthreads();
    });
    return tid ? type_t() : x;
  }

  MGPU_DEVICE type_t reduce(int tid, type_t x, storage_t& storage,
    op_t op = op_t()) const {
    return reduce(tid, x, nt, storage, op, false);
  }
};

#if __CUDA_ARCH__ >= 300

template<int nt, typename type_t, typename op_t>
struct cta_reduce_t<nt, type_t, op_t, true> {
  enum { 
    num_sections = warp_size, 
    section_size = nt / num_sections
  };
  struct storage_t { 
    type_t data[num_sections];
  };

  MGPU_DEVICE type_t reduce(int tid, type_t x, storage_t& storage, 
    int count = nt, op_t op = op_t()) const {

    if(tid >= count) x = 0;

    if(nt > warp_size) {
      int lane = (section_size - 1) & tid;
      int section = tid / section_size;

      // In the first phase, threads cooperatively reduce within their own
      // section.
      iterate<s_log2(section_size)>([&](int pass) {
        x = shfl_down_op(x, 1<< pass, plus_t<type_t>(), section_size);
      });

      // The last thread in each section stores the local reduction to shared 
      // memory.
      if(!lane)
        storage.data[section] = x;
      __syncthreads();

      // Reduce the totals of each input section.
      if(tid < num_sections) {
        x = storage.data[tid];
        iterate<s_log2(num_sections)>([&](int pass) {
          x = shfl_down_op(x, 1<< pass, plus_t<type_t>(), num_sections);
        });
      }
      __syncthreads();

    } else {
      iterate<s_log2(nt)>([&](int pass) {
        x = shfl_down_op(x, 1<< pass, plus_t<type_t>(), nt);
      });
    }

    return tid ? type_t() : x;
  }
};

#endif

END_MGPU_NAMESPACE
