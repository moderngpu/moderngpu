// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once
#include "tuple.hxx"

BEGIN_MGPU_NAMESPACE

///////////////////////
// tuple_iterator_value

template<typename tpl_t>
struct tuple_iterator_value;

template<typename... args_t>
struct tuple_iterator_value<tuple<args_t...> > {
  typedef tuple<typename std::iterator_traits<args_t>::value_type...> type;
};

template<typename tpl_t>
using tuple_iterator_value_t = typename tuple_iterator_value<tpl_t>::type;

////////////////////////////////////
// load and store to pointer tuples.

namespace detail {

template<typename int_t, typename... pointers_t, size_t... seq_i>
MGPU_HOST_DEVICE auto _lvalue_dereference(tuple<pointers_t...> pointers, 
  index_sequence<seq_i...> seq, int_t index) ->
  decltype(forward_as_tuple(get<seq_i>(pointers)[0]...)) {

  return forward_as_tuple(get<seq_i>(pointers)[index]...);
}

}

// Returns lvalues for each of the dereferenced pointers in the tuple.
template<typename int_t, typename... pointers_t>
MGPU_HOST_DEVICE auto dereference(tuple<pointers_t...> pointers, 
  int_t index) -> decltype(detail::_lvalue_dereference(pointers, 
    make_index_sequence<sizeof...(pointers_t)>(), index)) {

  return detail::_lvalue_dereference(pointers, 
    make_index_sequence<sizeof...(pointers_t)>(), index);
}

template<typename int_t, typename... pointers_t>
MGPU_HOST_DEVICE void store(tuple<pointers_t...> pointers, 
  tuple_iterator_value_t<tuple<pointers_t...> > values, 
  int_t index) {

  dereference(pointers, index) = values;
}

template<typename int_t, typename... pointers_t>
tuple_iterator_value_t<tuple<pointers_t...> > 
MGPU_HOST_DEVICE load(tuple<pointers_t...> pointers, int_t index) {
  typedef tuple_iterator_value_t<tuple<pointers_t...> > value_t;
  return value_t(dereference(pointers, index));
}

/////////////////////////////
// Tuple comparison operators

namespace detail {
template<size_t i, size_t count>
struct _tuple_compare {
  template<typename tpl_t>
  MGPU_HOST_DEVICE static bool eq(const tpl_t a, const tpl_t b) {
    return get<i>(a) == get<i>(b) && _tuple_compare<i + 1, count>::eq(a, b);
  }

  template<typename tpl_t>
  MGPU_HOST_DEVICE static bool less(const tpl_t a, const tpl_t b) {
    return get<i>(a) < get<i>(b) || 
      (!(get<i>(b) < get<i>(a)) && _tuple_compare<i + 1, count>::less(a, b));
  }
};

template<size_t count>
struct _tuple_compare<count, count> {
  template<typename tpl_t>
  MGPU_HOST_DEVICE static bool eq(const tpl_t, const tpl_t) {
    return true;
  }

  template<typename tpl_t>
  MGPU_HOST_DEVICE static bool less(const tpl_t, const tpl_t) {
    return false;
  }
};

} // namespace detail

//////////////////////////////////////////////
// Size of the largest component in the tuple.

template<size_t... values>
struct var_max;

template<size_t value_, size_t... values_> 
struct var_max<value_, values_...> {
  constexpr static size_t value = max(value_, var_max<values_...>::value);
};

template<size_t value_>
struct var_max<value_> {
  constexpr static size_t value = value_;
};

template<> struct var_max<> {
  constexpr static size_t value = 0;
};

template<typename tpl_t>
struct tuple_union_size;

template<typename... args_t>
struct tuple_union_size<tuple<args_t...> > {
  constexpr static size_t value = var_max<sizeof(args_t)...>::value;
};

END_MGPU_NAMESPACE

// Putting comparison operators back into global namespace.
template<typename... args_t>
MGPU_HOST_DEVICE bool operator<(const mgpu::tuple<args_t...>& a, 
  const mgpu::tuple<args_t...>& b) {
  return mgpu::detail::_tuple_compare<0, sizeof...(args_t)>::less(a, b);
}
template<typename... args_t>
MGPU_HOST_DEVICE bool operator<=(const mgpu::tuple<args_t...>& a, 
  const mgpu::tuple<args_t...>& b) {
  return !(b < a);
}
template<typename... args_t>
MGPU_HOST_DEVICE bool operator>(const mgpu::tuple<args_t...>& a, 
  const mgpu::tuple<args_t...>& b) {
  return b < a;
}
template<typename... args_t>
MGPU_HOST_DEVICE bool operator>=(const mgpu::tuple<args_t...>& a, 
  const mgpu::tuple<args_t...>& b) {
  return !(a < b);
}
template<typename... args_t>
MGPU_HOST_DEVICE bool operator==(const mgpu::tuple<args_t...>& a, 
  const mgpu::tuple<args_t...>& b) {
  return mgpu::detail::_tuple_compare<0, sizeof...(args_t)>::eq(a, b);
}
template<typename... args_t>
MGPU_HOST_DEVICE bool operator!=(const mgpu::tuple<args_t...>& a, 
  const mgpu::tuple<args_t...>& b) {
  return !(a == b);
}
