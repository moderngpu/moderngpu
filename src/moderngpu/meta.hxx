// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once

#include <typeinfo>
#include <type_traits>
#include <iterator>
#include <cassert>
#include <cfloat>
#include <cstdint>

#ifdef __CUDACC__

#ifndef MGPU_HOST_DEVICE
  #define MGPU_HOST_DEVICE __forceinline__ __device__ __host__
#endif

#ifndef MGPU_DEVICE 
  #define MGPU_DEVICE __device__
#endif

// Currently NVCC does not support __device__ __host__ tags on lambdas that
// are captured on the host and executed on the device. There is no good reason
// for this, as you can __device__ __host__ tag functor operators and use
// them in the same way. So for now, tag your functors with MGPU_LAMBDA. This
// means they are only supported in device code, but when a future version of
// CUDA lists this restriction MGPU_LAMBDA will be redefined to __device__
// __host__.
#ifndef MGPU_LAMBDA
  #define MGPU_LAMBDA __device__
#endif

#else // #ifndef __CUDACC__

#define MGPU_HOST_DEVICE

#endif // #ifdef __CUDACC__

#ifndef PRAGMA_UNROLL
#ifdef __CUDA_ARCH__
  #define PRAGMA_UNROLL #pragma PRAGMA_UNROLL
#else
  #define PRAGMA_UNROLL
#endif
#endif

#define BEGIN_MGPU_NAMESPACE namespace mgpu {
#define END_MGPU_NAMESPACE }

BEGIN_MGPU_NAMESPACE

template< bool B, class T = void >
using enable_if_t = typename std::enable_if<B,T>::type;

enum { warp_size = 32 };

#if defined(_MSC_VER) && _MSC_VER <= 1800      // VS 2013 is terrible.

#define is_pow2(x) (0 == ((x) & ((x) - 1)))
#define div_up(x, y) (((x) + (y) - 1) / (y))

namespace details {
template<int i, bool recurse = (i > 1)>
struct s_log2_t {
  enum { value = s_log2_t<i / 2>::value + 1 };
};
template<int i> struct s_log2_t<i, false> {
  enum { value = 0 };
};
} // namespace details

#define s_log2(x) details::s_log2_t<x>::value

#else

MGPU_HOST_DEVICE constexpr bool is_pow2(int x) {
  return 0 == (x & (x - 1));
}
MGPU_HOST_DEVICE constexpr int div_up(int x, int y) {
  return (x + y - 1) / y;
}
MGPU_HOST_DEVICE constexpr int64_t div_up(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}
MGPU_HOST_DEVICE constexpr size_t div_up(size_t x, size_t y) {
  return (x + y - 1) / y;
}
MGPU_HOST_DEVICE constexpr int s_log2(int x, int p = 0) {
  return x > 1 ? s_log2(x / 2) + 1 : p;
}
MGPU_HOST_DEVICE constexpr size_t s_log2(size_t x, size_t p = 0) {
  return x > 1 ? s_log2(x / 2) + 1 : p;
}

#endif

#ifdef _MSC_VER
  #define MGPU_ALIGN(x) __declspec(align(x))
#else
  #define MGPU_ALIGN(x) __attribute__((aligned(x)))
#endif

// Apparently not defined by CUDA.
template<typename real_t>
MGPU_HOST_DEVICE constexpr real_t min(real_t a, real_t b) {
  return (b < a) ? b : a;
}
template<typename real_t>
MGPU_HOST_DEVICE constexpr real_t max(real_t a, real_t b) {
  return (a < b) ? b : a;
}

struct empty_t { };

template<typename... args_t>
MGPU_HOST_DEVICE void swallow(args_t...) { }

template<typename... base_v>
struct inherit_t;

template<typename base_t, typename... base_v>
struct inherit_t<base_t, base_v...> : 
  base_t::template rebind<inherit_t<base_v...> > { };

template<typename base_t>
struct inherit_t<base_t> : base_t { };

////////////////////////////////////////////////////////////////////////////////
// Conditional typedefs. 

// Typedef type_a if type_a is not empty_t.
// Otherwise typedef type_b.
template<typename type_a, typename type_b>
struct conditional_typedef_t {
  typedef typename std::conditional<
    !std::is_same<type_a, empty_t>::value, 
    type_a, 
    type_b
  >::type type_t;
};

////////////////////////////////////////////////////////////////////////////////
// Code to treat __restrict__ as a CV qualifier.

template<typename arg_t>
struct is_restrict {
  enum { value = false };
};
template<typename arg_t>
struct is_restrict<arg_t __restrict__> {
  enum { value = true };
};

// Add __restrict__ only to pointers.
template<typename arg_t>
struct add_restrict {
  typedef arg_t type;
};
template<typename arg_t>
struct add_restrict<arg_t*> {
  typedef arg_t* __restrict__ type;
};

template<typename arg_t>
struct remove_restrict {
  typedef arg_t type;
};
template<typename arg_t>
struct remove_restrict<arg_t __restrict__> {
  typedef arg_t type;
};

template<typename arg_t>
MGPU_HOST_DEVICE typename add_restrict<arg_t>::type make_restrict(arg_t x) {
  typename add_restrict<arg_t>::type y = x;
  return y;
}

////////////////////////////////////////////////////////////////////////////////
// Template unrolled looping construct.

template<int i, int count, bool valid = (i < count)>
struct iterate_t {
  #pragma nv_exec_check_disable
  template<typename func_t>
  MGPU_HOST_DEVICE static void eval(func_t f) {
    f(i);
    iterate_t<i + 1, count>::eval(f);
  }
};
template<int i, int count>
struct iterate_t<i, count, false> {
  template<typename func_t>
  MGPU_HOST_DEVICE static void eval(func_t f) { }
};
template<int begin, int end, typename func_t>
MGPU_HOST_DEVICE void iterate(func_t f) {
  iterate_t<begin, end>::eval(f);
}
template<int count, typename func_t>
MGPU_HOST_DEVICE void iterate(func_t f) {
  iterate<0, count>(f);
}

template<int count, typename type_t>
MGPU_HOST_DEVICE type_t reduce(const type_t(&x)[count]) {
  type_t y;
  iterate<count>([&](int i) { y = i ? x[i] + y : x[i]; });
  return y;
}

template<int count, typename type_t>
MGPU_HOST_DEVICE void fill(type_t(&x)[count], type_t val) {
  iterate<count>([&](int i) { x[i] = val; });
}

#ifdef __CUDACC__

// Invoke unconditionally.
template<int nt, int vt, typename func_t>
MGPU_DEVICE void strided_iterate(func_t f, int tid) {
  iterate<vt>([=](int i) { f(i, nt * i + tid); });
}

// Check range.
template<int nt, int vt, int vt0 = vt, typename func_t>
MGPU_DEVICE void strided_iterate(func_t f, int tid, int count) {
  // Unroll the first vt0 elements of each thread.
  if(vt0 > 1 && count >= nt * vt0) {
    strided_iterate<nt, vt0>(f, tid);    // No checking
  } else {
    iterate<vt0>([=](int i) {
      int j = nt * i + tid;
      if(j < count) f(i, j);
    });
  }

  iterate<vt0, vt>([=](int i) {
    int j = nt * i + tid;
    if(j < count) f(i, j);
  });
}
template<int vt, typename func_t>
MGPU_DEVICE void thread_iterate(func_t f, int tid) {
  iterate<vt>([=](int i) { f(i, vt * tid + i); });
}

#endif // ifdef __CUDACC__

END_MGPU_NAMESPACE
