// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once

#include <random>
#include <algorithm>
#include "launch_box.hxx"

BEGIN_MGPU_NAMESPACE

// TODO: pass this state instead of tid/cta. This allows us to read in the
// count from device memory.
struct cta_state_t {
  int tid;
  int cta;
  size_t index;
  size_t count;
};

struct device_size_t {
  size_t host_size;
  const size_t* device_size;

  MGPU_HOST_DEVICE operator size_t() const { 
#ifdef __CUDA_ARCH__
    return *device_size; 
#else
    return host_size;
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
// Launch a grid given a number of CTAs.

template<typename launch_box, typename func_t>
void cta_launch(func_t f, int num_ctas, context_t& context) { 
  cta_dim_t cta = launch_box::cta_dim(context.ptx_version());
  launch_box_cta_k<launch_box, func_t>
    <<<num_ctas, cta.nt, 0, context.stream()>>>(f);
  context.synchronize();
}

template<int nt, typename func_t>
void cta_launch(func_t f, int num_ctas, context_t& context) {
  cta_launch<launch_params_t<nt, 1> >(f, num_ctas, context);
}

template<typename launch_box, typename func_t>
void cta_launch(func_t f, const int* num_ctas, context_t& context) {

  // TODO: Get number of CTAs that can be scheduled on the device.
  auto k = [=] MGPU_DEVICE(int tid, int cta) {
    const int* __restrict__ p_count = num_ctas;
    int count = *p_count;
    cta = blockIdx.x;
    while(cta < count) {
      f(tid, cta);
      cta += gridDim.x;
    }
  };
  cta_launch<launch_box>(k, 512, context);
}

////////////////////////////////////////////////////////////////////////////////
// Launch a grid given a number of work-items.

template<typename launch_box, typename func_t>
void cta_transform(func_t f, int count, context_t& context) {
  cta_dim_t cta = launch_box::cta_dim(context.ptx_version());
  int num_ctas = div_up(count, cta.nv());
  cta_launch<launch_box>(f, num_ctas, context);
}

template<int nt, int vt = 1, typename func_t>
void cta_transform(func_t f, int count, context_t& context) {
  cta_transform<launch_params_t<nt, vt> >(f, count, context);
}

////////////////////////////////////////////////////////////////////////////////
// Ordinary transform launch. It is preferable to make transform_k its own
// kernel rather than defining a lambda to convert (tid, cta) to index and 
// using cta_transform, because that approach results in extremely long
// mangled function names.

template<int nt, typename func_t>
__global__ void transform_k(func_t f, size_t count) {
  size_t index = nt * blockIdx.x + threadIdx.x;
  if(index < count) f(index);
}

template<size_t nt = 128, typename func_t>
void transform(func_t f, size_t count, context_t& context) {
  int num_ctas = (int)div_up(count, nt);
  transform_k<nt><<<num_ctas, nt, 0, context.stream()>>>(f, count);
}

END_MGPU_NAMESPACE
