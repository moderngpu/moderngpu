// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once
#include "loadstore.hxx"
#include "intrinsics.hxx"

BEGIN_MGPU_NAMESPACE

// cta_reduce_t returns the reduction of all inputs for thread 0, and returns
// type_t() for all other threads. This behavior saves a broadcast.

template<int nt, typename type_t>
struct cta_reduce_t {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300

  enum { 
    num_sections = warp_size, 
    section_size = nt / num_sections
  };
  struct storage_t { 
    type_t data[num_sections];
    type_t reduction;
  };

  template<typename op_t = plus_t<type_t> >
  MGPU_DEVICE type_t reduce(int tid, type_t x, storage_t& storage, 
    int count = nt, op_t op = op_t(), bool all_return = true) const {

    int lane = (section_size - 1) & tid;
    int section = tid / section_size;

    if(count >= nt) {
      // In the first phase, threads cooperatively reduce within their own
      // section.
      
      iterate<s_log2(section_size)>([&](int pass) {
        x = shfl_down_op(x, 1<< pass, op, section_size);
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
          x = shfl_down_op(x, 1<< pass, op, num_sections);
        });
        if(!tid && all_return) storage.reduction = x;
      }
      __syncthreads();
      
    } else {

      iterate<s_log2(section_size)>([&](int pass) {
        int offset = 1<< pass;
        type_t y = shfl_down(x, offset, section_size);
        if(tid < count - offset && lane < section_size - offset) 
          x = op(x, y);
      });
      if(!lane)
        storage.data[section] = x;
      __syncthreads();

      // Reduce the totals of each input section.
      if(tid < num_sections) {
        int spine_pop = div_up(count, section_size);
        x = storage.data[tid];
        iterate<s_log2(num_sections)>([&](int pass) {
          int offset = 1<< pass;
          type_t y = shfl_down(x, offset, num_sections);
          if(tid < spine_pop - offset) x = op(x, y);
        });
        if(!tid && all_return) storage.reduction = x;
      }
      __syncthreads();
    }

    if(all_return) {
      x = storage.reduction;
      __syncthreads();
    }

    return x;
  }

#else

  struct storage_t {
    type_t data[nt];
  };
  
  template<typename op_t = plus_t<type_t> >
  MGPU_DEVICE type_t reduce(int tid, type_t x, storage_t& storage, 
    int count = nt, op_t op = op_t(), bool all_return = true) const {

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

    if(all_return) {
      x = storage.data[0];
      __syncthreads();
    }

    return x;
  }

#endif  
};

END_MGPU_NAMESPACE
