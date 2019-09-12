// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once
#include "meta.hxx"

BEGIN_MGPU_NAMESPACE

namespace detail {

template<typename it_t, 
  typename type_t = typename std::iterator_traits<it_t>::value_type, 
  bool use_ldg = 
    std::is_pointer<it_t>::value && 
    std::is_arithmetic<type_t>::value
>
struct ldg_load_t {
  MGPU_HOST_DEVICE static type_t load(it_t it) {
    return *it;
  }
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350

template<typename it_t, typename type_t>
struct ldg_load_t<it_t, type_t, true> {
  MGPU_HOST_DEVICE static type_t load(it_t it) {
    return __ldg(it);
  }
};

#endif

} // namespace detail

template<typename it_t>
MGPU_HOST_DEVICE typename std::iterator_traits<it_t>::value_type
ldg(it_t it) {
  return detail::ldg_load_t<it_t>::load(it);
}

template<typename real_t>
MGPU_HOST_DEVICE real_t sq(real_t x) { return x * x; }

template<typename type_t>
MGPU_HOST_DEVICE void swap(type_t& a, type_t& b) {
  type_t c = a; a = b; b = c;
}

////////////////////////////////////////////////////////////////////////////////
// Device-side comparison operators.

template<typename type_t>
struct less_t : public std::binary_function<type_t, type_t, bool> {
  MGPU_HOST_DEVICE bool operator()(type_t a, type_t b) const {
    return a < b;
  }
};
template<typename type_t>
struct less_equal_t : public std::binary_function<type_t, type_t, bool> {
  MGPU_HOST_DEVICE bool operator()(type_t a, type_t b) const {
    return a <= b;
  }
};
template<typename type_t>
struct greater_t : public std::binary_function<type_t, type_t, bool> {
  MGPU_HOST_DEVICE bool operator()(type_t a, type_t b) const {
    return a > b;
  }
};
template<typename type_t>
struct greater_equal_t : public std::binary_function<type_t, type_t, bool> {
  MGPU_HOST_DEVICE bool operator()(type_t a, type_t b) const {
    return a >= b;
  }
};
template<typename type_t>
struct equal_to_t : public std::binary_function<type_t, type_t, bool> {
  MGPU_HOST_DEVICE bool operator()(type_t a, type_t b) const {
    return a == b;
  }
};
template<typename type_t>
struct not_equal_to_t : public std::binary_function<type_t, type_t, bool> {
  MGPU_HOST_DEVICE bool operator()(type_t a, type_t b) const {
    return a != b;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Device-side arithmetic operators.

template<typename type_t>
struct plus_t : public std::binary_function<type_t, type_t, type_t> {
	MGPU_HOST_DEVICE type_t operator()(type_t a, type_t b) const {
    return a + b;
  }
};

template<typename type_t>
struct minus_t : public std::binary_function<type_t, type_t, type_t> {
	MGPU_HOST_DEVICE type_t operator()(type_t a, type_t b) const {
    return a - b;
  }
};

template<typename type_t>
struct multiplies_t : public std::binary_function<type_t, type_t, type_t> {
  MGPU_HOST_DEVICE type_t operator()(type_t a, type_t b) const {
    return a * b;
  }
};

template<typename type_t>
struct maximum_t  : public std::binary_function<type_t, type_t, type_t> {
  MGPU_HOST_DEVICE type_t operator()(type_t a, type_t b) const {
    return max(a, b);
  }
};

template<typename type_t>
struct minimum_t  : public std::binary_function<type_t, type_t, type_t> {
  MGPU_HOST_DEVICE type_t operator()(type_t a, type_t b) const {
    return min(a, b);
  }
};

////////////////////////////////////////////////////////////////////////////////
// iterator_t and const_iterator_t are base classes for customized iterators.

template<typename outer_t, typename int_t, typename value_type>
struct iterator_t : public std::iterator_traits<const value_type*> {

  iterator_t() = default;
  MGPU_HOST_DEVICE iterator_t(int_t i) : index(i) { }

  MGPU_HOST_DEVICE outer_t operator+(int_t diff) const {
    outer_t next = *static_cast<const outer_t*>(this);
    next += diff;
    return next;
  }
  MGPU_HOST_DEVICE outer_t operator-(int_t diff) const {
    outer_t next = *static_cast<const outer_t*>(this);
    next -= diff;
    return next;
  }
  MGPU_HOST_DEVICE outer_t& operator+=(int_t diff) {
    index += diff;
    return *static_cast<outer_t*>(this);
  }
  MGPU_HOST_DEVICE outer_t& operator-=(int_t diff) {
    index -= diff;
    return *static_cast<outer_t*>(this);
  }

  int_t index;
};

template<typename outer_t, typename int_t, typename value_type>
struct const_iterator_t : public iterator_t<outer_t, int_t, value_type> {
  typedef iterator_t<outer_t, int_t, value_type> base_t;

  const_iterator_t() = default;
  MGPU_HOST_DEVICE const_iterator_t(int_t i) : base_t(i) { }

  // operator[] and operator* are tagged as DEVICE-ONLY.  This is to ensure
  // compatibility with lambda capture in CUDA 7.5, which does not support
  // marking a lambda as __host__ __device__.
  // We hope to relax this when a future CUDA fixes this problem.
  MGPU_HOST_DEVICE value_type operator[](int_t diff) const {
    return static_cast<const outer_t&>(*this)(base_t::index + diff);
  }
  MGPU_HOST_DEVICE value_type operator*() const {
    return (*this)[0];
  }
};

////////////////////////////////////////////////////////////////////////////////
// discard_iterator_t is a store iterator that discards its input.

template<typename value_type> 
struct discard_iterator_t : 
  iterator_t<discard_iterator_t<value_type>, int, value_type> {

  struct assign_t {
    MGPU_HOST_DEVICE value_type operator=(value_type v) { 
      return value_type(); 
    }
  };

  MGPU_HOST_DEVICE assign_t operator[](int index) const { 
    return assign_t(); 
  }
  MGPU_HOST_DEVICE assign_t operator*() const { return assign_t(); }
};

////////////////////////////////////////////////////////////////////////////////
// counting_iterator_t returns index.

template<typename type_t, typename int_t = int>
struct counting_iterator_t :
  const_iterator_t<counting_iterator_t<type_t>, int_t, type_t> {

  counting_iterator_t() = default;
  MGPU_HOST_DEVICE counting_iterator_t(type_t i) : 
    const_iterator_t<counting_iterator_t, int_t, type_t>(i) { }

  MGPU_HOST_DEVICE type_t operator()(int_t index) const {
    return (type_t)index;
  }
};

////////////////////////////////////////////////////////////////////////////////
// strided_iterator_t returns offset + index * stride.

template<typename type_t, typename int_t = int>
struct strided_iterator_t :
  const_iterator_t<strided_iterator_t<type_t>, int_t, int> {

  strided_iterator_t() = default;
  MGPU_HOST_DEVICE strided_iterator_t(type_t offset_, type_t stride_) : 
    const_iterator_t<strided_iterator_t, int_t, type_t>(0), 
    offset(offset_), stride(stride_) { }

  MGPU_HOST_DEVICE type_t operator()(int_t index) const {
    return offset + index * stride;
  }

  type_t offset, stride;
};

////////////////////////////////////////////////////////////////////////////////
// constant_iterator_t returns the value it was initialized with.

template<typename type_t>
struct constant_iterator_t : 
  const_iterator_t<constant_iterator_t<type_t>, int, type_t> {

  type_t value;

  MGPU_HOST_DEVICE constant_iterator_t(type_t value_) : value(value_) { }

  MGPU_HOST_DEVICE type_t operator()(int index) const {
    return value;
  }
};

// These types only supported with nvcc until CUDA 8.0 allows host-device
// lambdas and MGPU_LAMBDA is redefined to MGPU_HOST_DEVICE

#ifdef __CUDACC__

////////////////////////////////////////////////////////////////////////////////
// lambda_iterator_t

template<typename load_t, typename store_t, typename value_type, typename int_t>
struct lambda_iterator_t : std::iterator_traits<const value_type*> {

  load_t load;
  store_t store;
  int_t base;

  lambda_iterator_t(load_t load_, store_t store_, int_t base_) :
    load(load_), store(store_), base(base_) { }

  struct assign_t {
    load_t load;
    store_t store;
    int_t index;

    MGPU_LAMBDA assign_t& operator=(value_type rhs) {
      static_assert(!std::is_same<store_t, empty_t>::value, 
        "load_iterator is being stored to.");
      store(rhs, index);
      return *this;
    }
    MGPU_LAMBDA operator value_type() const {
      static_assert(!std::is_same<load_t, empty_t>::value,
        "store_iterator is being loaded from.");
      return load(index);
    }
  };

  MGPU_LAMBDA assign_t operator[](int_t index) const {
    return assign_t { load, store, base + index };
  } 
  MGPU_LAMBDA assign_t operator*() const {
    return assign_t { load, store, base };
  }

  MGPU_HOST_DEVICE lambda_iterator_t operator+(int_t offset) const {
    lambda_iterator_t cp = *this;
    cp += offset;
    return cp;
  }

  MGPU_HOST_DEVICE lambda_iterator_t& operator+=(int_t offset) {
    base += offset;
    return *this;
  }

  MGPU_HOST_DEVICE lambda_iterator_t operator-(int_t offset) const {
    lambda_iterator_t cp = *this;
    cp -= offset;
    return cp;
  }

  MGPU_HOST_DEVICE lambda_iterator_t& operator-=(int_t offset) {
    base -= offset;
    return *this;
  }
};

template<typename value_type>
struct trivial_load_functor {
  template<typename int_t>
  MGPU_HOST_DEVICE value_type operator()(int_t index) const {
    return value_type();
  }
};

template<typename value_type>
struct trivial_store_functor {
  template<typename int_t>
  MGPU_HOST_DEVICE void operator()(value_type v, int_t index) const { }
};

template<typename value_type, typename int_t = int, typename load_t, 
  typename store_t>
lambda_iterator_t<load_t, store_t, value_type, int_t> 
  make_load_store_iterator(load_t load, store_t store, int_t base = 0) {
  return lambda_iterator_t<load_t, store_t, value_type, int_t>(load, store, base);
}

template<typename value_type, typename int_t = int, typename load_t>
lambda_iterator_t<load_t, empty_t, value_type, int_t>
make_load_iterator(load_t load, int_t base = 0) {
  return make_load_store_iterator<value_type>(load, empty_t(), base);
}

template<typename value_type, typename int_t = int, typename store_t>
lambda_iterator_t<empty_t, store_t, value_type, int_t>
make_store_iterator(store_t store, int_t base = 0) {
  return make_load_store_iterator<value_type>(empty_t(), store, base);
}

#endif // #ifdef __CUDACC__

END_MGPU_NAMESPACE
