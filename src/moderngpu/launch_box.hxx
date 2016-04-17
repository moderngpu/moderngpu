// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once

#include "context.hxx"

BEGIN_MGPU_NAMESPACE

// Specializable launch parameters.
struct launch_box_default_t {
  typedef launch_cta_t<0, 0, 0> sm_00;
  typedef empty_t sm_20, sm_21, sm_30, sm_32, sm_35, sm_37, sm_50, sm_52, sm_53;

  template<typename new_base_t>
  using rebind = launch_box_default_t;
};

template<typename... params_v>
struct launch_box_t : inherit_t<params_v..., launch_box_default_t> { 
  typedef inherit_t<params_v..., launch_box_default_t> base_t; 

  typedef typename conditional_typedef_t<
    typename base_t::sm_20, typename base_t::sm_00
  >::type_t sm_20;

#define INHERIT_LAUNCH_PARAMS(new_ver, old_ver) \
  typedef typename conditional_typedef_t< \
    typename base_t::sm_##new_ver, sm_##old_ver \
  >::type_t sm_##new_ver;
  
  INHERIT_LAUNCH_PARAMS(21, 20)
  INHERIT_LAUNCH_PARAMS(30, 21)
  INHERIT_LAUNCH_PARAMS(32, 30)
  INHERIT_LAUNCH_PARAMS(35, 30)
  INHERIT_LAUNCH_PARAMS(37, 35)
  INHERIT_LAUNCH_PARAMS(50, 35)
  INHERIT_LAUNCH_PARAMS(52, 50)
  INHERIT_LAUNCH_PARAMS(53, 50)

  // Overwrite the params defined for sm_00 so that the host-side compiler
  // has all expected symbols available to it.
  typedef sm_53 sm_00;
  typedef MGPU_LAUNCH_PARAMS(launch_box_t) sm_ptx;

  static cta_dim_t cta_dim(int ptx_version) {
    // Ptx version from cudaFuncGetAttributes.
    if     (ptx_version == 53) return cta_dim_t { sm_53::nt, sm_53::vt };
    else if(ptx_version >= 52) return cta_dim_t { sm_52::nt, sm_52::vt };
    else if(ptx_version >= 50) return cta_dim_t { sm_50::nt, sm_50::vt };
    else if(ptx_version == 37) return cta_dim_t { sm_37::nt, sm_37::vt };
    else if(ptx_version >= 35) return cta_dim_t { sm_35::nt, sm_35::vt };
    else if(ptx_version == 32) return cta_dim_t { sm_32::nt, sm_32::vt };
    else if(ptx_version >= 30) return cta_dim_t { sm_30::nt, sm_30::vt };
    else if(ptx_version >= 21) return cta_dim_t { sm_21::nt, sm_21::vt };
    else if(ptx_version >= 20) return cta_dim_t { sm_20::nt, sm_20::vt };
    else return cta_dim_t { -1, 0 };
  }

  static cta_dim_t cta_dim(const context_t& context) {
    return cta_dim(context.ptx_version());
  }

  static int nv(const context_t& context) {
    return cta_dim(context.ptx_version()).nv();
  }
};


template<typename launch_box, typename func_t, typename... args_t>
int occupancy(func_t f, const context_t& context, args_t... args) {
  int num_blocks;
  int nt = launch_box::cta_dim(context).nt;
  cudaError_t result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks, 
    &launch_box_cta_k<launch_box, func_t, args_t...>, 
    nt,
    (size_t)0
  );
  if(cudaSuccess != result) throw cuda_exception_t(result);
  return context.props().multiProcessorCount * num_blocks;
}

END_MGPU_NAMESPACE
