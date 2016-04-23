// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once

#include "meta.hxx"
#include "tuple.hxx"

#if   __CUDA_ARCH__ == 530
  #define MGPU_SM_TAG sm_53
#elif __CUDA_ARCH__ >= 520
  #define MGPU_SM_TAG sm_52
#elif __CUDA_ARCH__ >= 500
  #define MGPU_SM_TAG sm_50
#elif __CUDA_ARCH__ == 370
  #define MGPU_SM_TAG sm_37
#elif __CUDA_ARCH__ >= 350
  #define MGPU_SM_TAG sm_35
#elif __CUDA_ARCH__ == 320
  #define MGPU_SM_TAG sm_32
#elif __CUDA_ARCH__ >= 300
  #define MGPU_SM_TAG sm_30
#elif __CUDA_ARCH__ >= 210
  #define MGPU_SM_TAG sm_21
#elif __CUDA_ARCH__ >= 200
  #define MGPU_SM_TAG sm_20
#elif defined(__CUDA_ARCH__)
  #error "Modern GPU v3 does not support builds for sm_1.x"
#else
  #define MGPU_SM_TAG sm_00
#endif

#define MGPU_LAUNCH_PARAMS(launch_box) \
  typename launch_box::MGPU_SM_TAG
#define MGPU_LAUNCH_BOUNDS(launch_box) \
  __launch_bounds__(launch_box::sm_ptx::nt, launch_box::sm_ptx::occ) 

BEGIN_MGPU_NAMESPACE

struct MGPU_ALIGN(8) cta_dim_t {
  int nt, vt;
  int nv() const { return nt * vt; }
  int num_ctas(int count) const {
    return div_up(count, nv());
  }
};

// Generic thread cta kernel.
template<typename launch_box, typename func_t, typename... args_t>
__global__ MGPU_LAUNCH_BOUNDS(launch_box)
void launch_box_cta_k(func_t f, int num_ctas, args_t... args) {
  // Masking threadIdx.x by (nt - 1) may help strength reduction because the
  // compiler now knows the range of tid: (0, nt).
  typedef typename launch_box::sm_ptx params_t;
  int tid = (params_t::nt - 1) & threadIdx.x;
  int cta = blockIdx.x;

  // Convert the arguments to restricted pointer types.
  auto restricted_args = restrict_tuple(tuple<args_t...>(args...));

#if __CUDA_ARCH__ < 300
  cta += gridDim.x * blockIdx.y;
  if(cta < num_ctas)
#endif
    tuple_expand(f, tuple_cat(make_tuple(tid, cta), restricted_args));
}

// Dummy kernel for retrieving PTX version.
template<int dummy_arg>
__global__ void dummy_k() { }

template<int nt_, int vt_ = 1, int vt0_ = vt_, int occ_= 0>
struct launch_cta_t {
  enum { nt = nt_, vt = vt_, vt0 = vt0_, occ = occ_ };
};

#define DEF_ARCH_STRUCT(ver) \
  template<typename params_t, typename base_t = empty_t> \
  struct arch_##ver : base_t { \
    typedef params_t sm_##ver; \
 \
    template<typename new_base_t> \
    using rebind = arch_##ver<params_t, new_base_t>; \
  }; \
  \
  template<int nt, int vt = 1, int vt0 = vt, int occ = 0> \
  using arch_##ver##_cta = arch_##ver<launch_cta_t<nt, vt, vt0, occ> >;

DEF_ARCH_STRUCT(20)
DEF_ARCH_STRUCT(21)
DEF_ARCH_STRUCT(30)
DEF_ARCH_STRUCT(32)
DEF_ARCH_STRUCT(35)
DEF_ARCH_STRUCT(37)
DEF_ARCH_STRUCT(50)
DEF_ARCH_STRUCT(52)
DEF_ARCH_STRUCT(53)

#undef DEF_ARCH_STRUCT

struct context_t;

// Non-specializable launch parameters.
template<int nt, int vt, int vt0 = vt, int occ = 0>
struct launch_params_t : launch_cta_t<nt, vt, vt0, occ> {
  typedef launch_params_t sm_ptx;

  static cta_dim_t cta_dim() {
    return cta_dim_t { nt, vt };
  }

  static cta_dim_t cta_dim(int) {
    return cta_dim();
  }

  static cta_dim_t cta_dim(const context_t& context) {
    return cta_dim();
  }

  static int nv(const context_t& context) {
    return cta_dim().nv();
  }
};

END_MGPU_NAMESPACE
