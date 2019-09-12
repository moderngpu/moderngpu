// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once

#include "transform.hxx"
#include "context.hxx"

BEGIN_MGPU_NAMESPACE

////////////////////////////////////////////////////////////////////////////////
// Memory functions on raw pointers.

template<typename type_t>
cudaError_t htoh(type_t* dest, const type_t* source, size_t count) {
  if(count) 
    memcpy(dest, source, sizeof(type_t) * count);
  return cudaSuccess;
}

template<typename type_t>
cudaError_t dtoh(type_t* dest, const type_t* source, size_t count) {
  cudaError_t result = count ? 
    cudaMemcpy(dest, source, sizeof(type_t) * count,
      cudaMemcpyDeviceToHost) :
    cudaSuccess;
  return result;
}

template<typename type_t>
cudaError_t htod(type_t* dest, const type_t* source, size_t count) {
  cudaError_t result = count ?
    cudaMemcpy(dest, source, sizeof(type_t) * count,
      cudaMemcpyHostToDevice) :
    cudaSuccess;
  return result;
}

template<typename type_t>
cudaError_t dtod(type_t* dest, const type_t* source, size_t count) {
  cudaError_t result = count ?
    cudaMemcpy(dest, source, sizeof(type_t) * count,
      cudaMemcpyDeviceToDevice) :
    cudaSuccess;
  return result;
}

template<typename type_t>
cudaError_t dtoh(std::vector<type_t>& dest, const type_t* source, 
  size_t count) {
  dest.resize(count);
  return dtoh(dest.data(), source, count);
}

template<typename type_t>
cudaError_t htod(type_t* dest, const std::vector<type_t>& source) {
  return htod(dest, source.data(), source.size());
}

////////////////////////////////////////////////////////////////////////////////
// Memory functions on mem_t.

template<typename type_t>
mem_t<type_t> to_mem(const std::vector<type_t>& data, context_t& context) {
  mem_t<type_t> mem(data.size(), context);
  cudaError_t result = htod(mem.data(), data);
  if(cudaSuccess != result) throw cuda_exception_t(result);
  return mem;
}

template<typename type_t>
std::vector<type_t> from_mem(const mem_t<type_t>& mem) {
  std::vector<type_t> host;
  cudaError_t result = dtoh(host, mem.data(), mem.size());
  if(cudaSuccess != result) throw cuda_exception_t(result);
  return host;
}

template<typename type_t, typename func_t>
mem_t<type_t> fill_function(func_t f, size_t count, context_t& context) {
  mem_t<type_t> mem(count, context);
  type_t* p = mem.data();
  transform([=]MGPU_DEVICE(int index) {
    p[index] = f(index);
  }, count, context);
  return mem;
}

template<typename type_t>
mem_t<type_t> fill(type_t value, size_t count, context_t& context) {
  // We'd prefer to call fill_function and pass a lambda that returns value,
  // but that can create tokens that are too long for VS2013.
  mem_t<type_t> mem(count, context);
  type_t* p = mem.data();
  transform([=]MGPU_DEVICE(int index) {
    p[index] = value;
  }, count, context);
  return mem;
}

template<typename it_t>
auto copy_to_mem(it_t input, size_t count, context_t& context) -> 
  mem_t<typename std::iterator_traits<it_t>::value_type> {
  
  typedef typename std::iterator_traits<it_t>::value_type type_t;
  mem_t<type_t> mem(count, context);
  type_t* p = mem.data();
  transform([=]MGPU_DEVICE(int index) {
    p[index] = input[index];
  }, count, context);
  return mem;
}

inline std::mt19937& get_mt19937() {
  static std::mt19937 mt19937;
  return mt19937;
}

mem_t<int> inline fill_random(int a, int b, size_t count, bool sorted, 
  context_t& context) {

  std::uniform_int_distribution<int> d(a, b);
  std::vector<int> data(count);

  for(int& i : data)
    i = d(get_mt19937());
  if(sorted) 
    std::sort(data.begin(), data.end());

  return to_mem(data, context);
}

END_MGPU_NAMESPACE
