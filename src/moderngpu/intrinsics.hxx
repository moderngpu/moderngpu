// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once

#include "operators.hxx"

#ifndef __CUDACC__
#error "You must compile this file with nvcc. You must."
#endif

BEGIN_MGPU_NAMESPACE

////////////////////////////////////////////////////////////////////////////////
// brev, popc, clz, bfe, bfi, prmt

// Reverse the bits in an integer.
MGPU_HOST_DEVICE unsigned brev(unsigned x) { 
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
  unsigned y = __brev(x);
#else
  unsigned y = 0;
  for(int i = 0; i < 32; ++i)
    y |= (1 & (x>> i))<< (31 - i);
#endif
  return y;
}

// Count number of bits in a register.
MGPU_HOST_DEVICE int popc(unsigned x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
  return __popc(x);
#else
  int c;
  for(c = 0; x; ++c)
    x &= x - 1;
  return c;
#endif
}

// Count leading zeros - start from most significant bit.
MGPU_HOST_DEVICE int clz(int x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
  return __clz(x);
#else
  for(int i = 31; i >= 0; --i)
    if((1<< i) & x) return 31 - i;
  return 32;
#endif
}

// Find first set - start from least significant bit. LSB is 1. ffs(0) is 0.
MGPU_HOST_DEVICE int ffs(int x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
  return __ffs(x);
#else
  for(int i = 0; i < 32; ++i)
    if((1<< i) & x) return i + 1;
  return 0;
#endif
}

MGPU_HOST_DEVICE unsigned bfe(unsigned x, unsigned bit, unsigned num_bits) {
  unsigned result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
  asm("bfe.u32 %0, %1, %2, %3;" : 
    "=r"(result) : "r"(x), "r"(bit), "r"(num_bits));
#else
  result = ((1<< num_bits) - 1) & (x>> bit);
#endif
  return result;
}

MGPU_HOST_DEVICE unsigned bfi(unsigned x, unsigned y, unsigned bit, 
  unsigned num_bits) {
  unsigned result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
  asm("bfi.b32 %0, %1, %2, %3, %4;" : 
    "=r"(result) : "r"(x), "r"(y), "r"(bit), "r"(num_bits));
#else
  if(bit + num_bits > 32) num_bits = 32 - bit;
  unsigned mask = ((1<< num_bits) - 1)<< bit;
  result = y & ~mask;
  result |= mask & (x<< bit);
#endif
  return result;
}

MGPU_HOST_DEVICE unsigned prmt(unsigned a, unsigned b, unsigned index) {
  unsigned result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(index));
#else
  result = 0;
  for(int i = 0; i < 4; ++i) {
    unsigned sel = 0xf & (index>> (4 * i));
    unsigned x = ((7 & sel) > 3) ? b : a;
    x = 0xff & (x>> (8 * (3 & sel)));
    if(8 & sel) x = (128 & x) ? 0xff : 0;
    result |= x<< (8 * i);
  }
#endif
  return result;
}

// Find log2(x) and optionally round up to the next integer logarithm.
MGPU_HOST_DEVICE int find_log2(int x, bool round_up = false) {
  int a = 31 - clz(x);
  if(round_up) a += !is_pow2(x);
  return a;
} 

////////////////////////////////////////////////////////////////////////////////
// Divide operators.

MGPU_HOST_DEVICE int mulhi(int a, int b) {
#ifdef __CUDA_ARCH__
  return __mulhi(a, b);
#else
  union {
    int64_t x;
    struct { int low, high; };
  } product;
  product.x = (int64_t)a * b;
  return product.high;
#endif
}

MGPU_HOST_DEVICE unsigned umulhi(unsigned a, unsigned b) {
#ifdef __CUDA_ARCH__
  return __mulhi(a, b);
#else
  union {
    uint64_t x;
    struct { unsigned low, high; };
  } product;
  product.x = (uint64_t)a * b;
  return product.high; 
#endif  
}

////////////////////////////////////////////////////////////////////////////////
// Wrappers around PTX shfl_up and shfl_down.

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300

template<typename type_t>
MGPU_DEVICE type_t shfl_up(type_t x, int offset, int width = warp_size) { 
  enum { num_words = div_up(sizeof(type_t), sizeof(int)) };
  union {
    int x[num_words];
    type_t t;
  } u;
  u.t = x;

  iterate<num_words>([&](int i) {
    u.x[i] = __shfl_up(u.x[i], offset, width);
  });
  return u.t;
}

template<typename type_t>
MGPU_DEVICE type_t shfl_down(type_t x, int offset, int width = warp_size) { 
  enum { num_words = div_up(sizeof(type_t), sizeof(int)) };
  union {
    int x[num_words];
    type_t t;
  } u;
  u.t = x;

  iterate<num_words>([&](int i) {
    u.x[i] = __shfl_down(u.x[i], offset, width);
  });
  return u.t;
}

template<typename type_t, typename op_t> 
MGPU_DEVICE type_t shfl_up_op(type_t x, int offset, op_t op, 
  int width = warp_size) {

  type_t y = shfl_up(x, offset, width);
  int lane = (width - 1) & threadIdx.x;
  if(lane >= offset) x = op(x, y);
  return x;
}

template<typename type_t, typename op_t> 
MGPU_DEVICE type_t shfl_down_op(type_t x, int offset, op_t op, 
  int width = warp_size) {

  type_t y = shfl_down(x, offset, width);
  int lane = (width - 1) & threadIdx.x;
  if(lane < width - offset) x = op(x, y);
  return x;
}

#define SHFL_OP_MACRO(dir, is_up, ptx_type, r, c_type, ptx_op, c_op) \
MGPU_DEVICE inline c_type shfl_##dir##_op(c_type x, int offset, \
  c_op<c_type> op, int width = warp_size) { \
  c_type result = c_type(); \
  int mask = (warp_size - width)<< 8 | (is_up ? 0 : (width - 1)); \
  asm( \
    "{.reg ."#ptx_type" r0;" \
    ".reg .pred p;" \
    "shfl."#dir".b32 r0|p, %1, %2, %3;" \
    "@p "#ptx_op"."#ptx_type" r0, r0, %4;" \
    "mov."#ptx_type" %0, r0; }" \
    : "="#r(result) : #r(x), "r"(offset), "r"(mask), #r(x)); \
  return result; \
}

SHFL_OP_MACRO(up, true, s32, r, int, add, plus_t)
SHFL_OP_MACRO(up, true, s32, r, int, max, maximum_t)
SHFL_OP_MACRO(up, true, s32, r, int, min, minimum_t)
SHFL_OP_MACRO(down, false, s32, r, int, add, plus_t)
SHFL_OP_MACRO(down, false, s32, r, int, max, maximum_t)
SHFL_OP_MACRO(down, false, s32, r, int, min, minimum_t)

SHFL_OP_MACRO(up, true, u32, r, unsigned, add, plus_t)
SHFL_OP_MACRO(up, true, u32, r, unsigned, max, maximum_t)
SHFL_OP_MACRO(up, true, u32, r, unsigned, min, minimum_t)
SHFL_OP_MACRO(down, false, u32, r, unsigned, add, plus_t)
SHFL_OP_MACRO(down, false, u32, r, unsigned, max, maximum_t)
SHFL_OP_MACRO(down, false, u32, r, unsigned, min, minimum_t)

SHFL_OP_MACRO(up, true, f32, f, float, add, plus_t)
SHFL_OP_MACRO(up, true, f32, f, float, max, maximum_t)
SHFL_OP_MACRO(up, true, f32, f, float, max, minimum_t)
SHFL_OP_MACRO(down, false, f32, f, float, add, plus_t)
SHFL_OP_MACRO(down, false, f32, f, float, max, maximum_t)
SHFL_OP_MACRO(down, false, f32, f, float, max, minimum_t)

#undef SHFL_OP_MACRO

#define SHFL_OP_64b_MACRO(dir, is_up, ptx_type, r, c_type, ptx_op, c_op) \
MGPU_DEVICE inline c_type shfl_##dir##_op(c_type x, int offset, \
  c_op<c_type> op, int width = warp_size) { \
  c_type result = c_type(); \
  int mask = (warp_size - width)<< 8 | (is_up ? 0 : (width - 1)); \
  asm( \
    "{.reg ."#ptx_type" r0;" \
    ".reg .u32 lo;" \
    ".reg .u32 hi;" \
    ".reg .pred p;" \
    "mov.b64 {lo, hi}, %1;" \
    "shfl."#dir".b32 lo|p, lo, %2, %3;" \
    "shfl."#dir".b32 hi  , hi, %2, %3;" \
    "mov.b64 r0, {lo, hi};" \
    "@p "#ptx_op"."#ptx_type" r0, r0, %4;" \
    "mov."#ptx_type" %0, r0; }" \
    : "="#r(result) : #r(x), "r"(offset), "r"(mask), #r(x) \
  ); \
  return result; \
}

SHFL_OP_64b_MACRO(up, true, s64, l, int64_t, add, plus_t)
SHFL_OP_64b_MACRO(up, true, s64, l, int64_t, max, maximum_t)
SHFL_OP_64b_MACRO(up, true, s64, l, int64_t, min, minimum_t)
SHFL_OP_64b_MACRO(down, false, s64, l, int64_t, add, plus_t)
SHFL_OP_64b_MACRO(down, false, s64, l, int64_t, max, maximum_t)
SHFL_OP_64b_MACRO(down, false, s64, l, int64_t, min, minimum_t)

SHFL_OP_64b_MACRO(up, true, u64, l, uint64_t, add, plus_t)
SHFL_OP_64b_MACRO(up, true, u64, l, uint64_t, max, maximum_t)
SHFL_OP_64b_MACRO(up, true, u64, l, uint64_t, min, minimum_t)
SHFL_OP_64b_MACRO(down, false, u64, l, uint64_t, add, plus_t)
SHFL_OP_64b_MACRO(down, false, u64, l, uint64_t, max, maximum_t)
SHFL_OP_64b_MACRO(down, false, u64, l, uint64_t, min, minimum_t)

SHFL_OP_64b_MACRO(up, true, f64, d, double, add, plus_t)
SHFL_OP_64b_MACRO(up, true, f64, d, double, max, maximum_t)
SHFL_OP_64b_MACRO(up, true, f64, d, double, min, minimum_t)
SHFL_OP_64b_MACRO(down, false, f64, d, double, add, plus_t)
SHFL_OP_64b_MACRO(down, false, f64, d, double, max, maximum_t)
SHFL_OP_64b_MACRO(down, false, f64, d, double, min, minimum_t)

#undef SHFL_OP_64b_MACRO

#endif

END_MGPU_NAMESPACE
