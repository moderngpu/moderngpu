// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once
#include "meta.hxx"
#include "operators.hxx"

// A barebones tuple implementation for CUDA.

BEGIN_MGPU_NAMESPACE

// tuple
template<typename... args_t>
struct tuple;

template<> struct tuple<> { };

template<typename arg_t, typename... args_t>
struct tuple<arg_t, args_t...> : tuple<args_t...> {
  typedef tuple<args_t...> inner_t;
  arg_t x;

  tuple() = default;
  MGPU_HOST_DEVICE tuple(arg_t arg, args_t... args) : 
    x(arg), inner_t(args...) { }
};

template<typename arg_t>
struct tuple<arg_t> : tuple<> {
  typedef tuple<> inner_t;
  arg_t x;

  tuple() = default;
  MGPU_HOST_DEVICE tuple(arg_t arg) : x(arg) { }
};

// tuple_size
template<typename... args_t>
struct tuple_size;

template<typename... args_t>
struct tuple_size<tuple<args_t...> > {
  enum { value = sizeof...(args_t) };
};

// tuple_element
template<int i, typename... args_t>
struct tuple_element;

template<int i, typename arg_t, typename... args_t>
struct tuple_element<i, tuple<arg_t, args_t...> > {
  typedef typename tuple_element<i - 1, tuple<args_t...> >::type type;
};
template<typename arg_t, typename... args_t>
struct tuple_element<0, tuple<arg_t, args_t...> > {
  typedef arg_t type;
};
template<typename arg_t>
struct tuple_element<0, tuple<arg_t> > {
  typedef arg_t type;
};

// get
namespace detail {

template<int i, typename... args_t>
struct get_tuple_t;

template<int i, typename arg_t, typename... args_t>
struct get_tuple_t<i, tuple<arg_t, args_t...> > {
  typedef typename tuple_element<i, tuple<arg_t, args_t...> >::type type_t;
  typedef tuple<args_t...> inner_tuple_t;
  typedef typename inner_tuple_t::inner_t inner_t;

  MGPU_HOST_DEVICE static type_t& get(tuple<arg_t, args_t...>& t) {
    return get_tuple_t<i - 1, inner_tuple_t>::get(t);
  }
  MGPU_HOST_DEVICE static type_t&& get(tuple<arg_t, args_t...>&& t) {
    return get_tuple_t<i - 1, inner_tuple_t>::get(mgpu::forward<inner_t>(t));
  }
  MGPU_HOST_DEVICE static const type_t& get(const tuple<arg_t, args_t...>& t) {
    return get_tuple_t<i - 1, inner_tuple_t>::get(t);
  }
};
template<typename arg_t, typename... args_t>
struct get_tuple_t<0, tuple<arg_t, args_t...> > {
  MGPU_HOST_DEVICE static arg_t& get(tuple<arg_t, args_t...>& t) {
    return t.x;
  }
  MGPU_HOST_DEVICE static arg_t&& get(tuple<arg_t, args_t...>&& t) {
    return t.x;
  }
  MGPU_HOST_DEVICE static const arg_t& get(const tuple<arg_t, args_t...>& t) {
    return t.x;
  }
};
template<typename arg_t>
struct get_tuple_t<0, tuple<arg_t> > {
  MGPU_HOST_DEVICE static arg_t& get(tuple<arg_t>& t) {
    return t.x;
  }
  MGPU_HOST_DEVICE static arg_t&& get(tuple<arg_t>&& t) {
    return t.x;
  }
  MGPU_HOST_DEVICE static const arg_t& get(const tuple<arg_t>& t) {
    return t.x;
  }
};

}

template<int i, typename... args_t>
MGPU_HOST_DEVICE typename tuple_element<i, tuple<args_t...> >::type&
get(tuple<args_t...>& t) {
  return detail::get_tuple_t<i, tuple<args_t...> >::get(t);
}
template<int i, typename... args_t>
MGPU_HOST_DEVICE typename tuple_element<i, tuple<args_t...> >::type&&
get(tuple<args_t...>&& t) {
  return detail::get_tuple_t<i, tuple<args_t...> >::get(
    mgpu::forward<tuple<args_t...> >(t)
  );
}
template<int i, typename... args_t>
MGPU_HOST_DEVICE const typename tuple_element<i, tuple<args_t...> >::type&
get(const tuple<args_t...>& t) {
  return detail::get_tuple_t<i, tuple<args_t...> >::get(t);
}

// make_tuple
template<typename... args_t>
MGPU_HOST_DEVICE tuple<args_t...> make_tuple(args_t... args) {
  return tuple<args_t...>(args...);
}

// combine_tuples_t
template<typename...>
struct combine_tuples_t;

template<> 
struct combine_tuples_t<> {
  typedef tuple<> type;
};
template<typename... args_t>
struct combine_tuples_t<tuple<args_t...> > {
  typedef tuple<args_t...> type;
};
template<typename... a_t, typename... b_t, typename... c_t>
struct combine_tuples_t<tuple<a_t...>, tuple<b_t...>, c_t...> {
  typedef typename combine_tuples_t<tuple<a_t..., b_t...>, c_t...>::type type;
};

////////////////////////////////////////////////////////////////////////////////
// Element-by-element type conversion of one tuple into another tuple.

template<template<typename> class convert_t, typename... args_t>
struct tuple_convert_t;

template<template<typename> class convert_t, 
  typename arg_t, typename... args_t>
struct tuple_convert_t<convert_t, tuple<arg_t, args_t...> > {
  typedef typename convert_t<arg_t>::value_type new_t;
  typedef typename tuple_convert_t<convert_t, tuple<args_t...> >::type_t rhs_t;
  typedef typename combine_tuples_t<tuple<new_t>, rhs_t>::type type_t;
};

template<template<typename> class convert_t, typename arg_t>
struct tuple_convert_t<convert_t, tuple<arg_t> > {
  typedef typename convert_t<arg_t>::value_type new_t;
  typedef tuple<new_t> type_t;
};

template<template<typename> class convert_t>
struct tuple_convert_t<convert_t, tuple<> > {
  typedef tuple<> type_t;
};

template<int i, int j, int count>
struct tuple_iterate_assign_t {
  template<typename assign_t, typename input_t, typename output_t, 
    typename... args_t>
  MGPU_HOST_DEVICE static void assign(assign_t a, input_t input, 
    output_t& output, args_t... args) {
    typename tuple_element<i, input_t>::type lhs = get<i>(input);
    typename tuple_element<i, output_t>::type& rhs = get<j>(output);
    a(lhs, rhs, args...);
    tuple_iterate_assign_t<i + 1, j + 1, count - 1>::assign(a, input, output,
      args...);
  }
};
template<int i, int j>
struct tuple_iterate_assign_t<i, j, 0> {
  template<typename assign_t, typename input_t, typename output_t, 
    typename... args_t>
  MGPU_HOST_DEVICE static void assign(assign_t, input_t, output_t&, 
    args_t...) { }  
};

struct assign_t {
  template<typename input_it, typename output_it>
  MGPU_HOST_DEVICE void operator()(input_it input, output_it& output) {
    output = input;
  }
};

////////////////////////////////////////////////////////////////////////////////
// tuple_cat. Glue together any number of tuples.

namespace detail {

template<typename... args_t>
struct tuple_cat_t;

template<>
struct tuple_cat_t<tuple<> > {
  typedef tuple<> type_t;
  MGPU_HOST_DEVICE static tuple<> cat(tuple<>) { return tuple<>(); }
};

template<typename... args_t>
struct tuple_cat_t<tuple<args_t...> > {
  MGPU_HOST_DEVICE static tuple<args_t...> 
    cat(tuple<args_t...> tpl) { return tpl; }
};

template<typename... lhs_t, typename... rhs_t>
struct tuple_cat_t<tuple<lhs_t...>, tuple<rhs_t...> > {
  enum { len1 = sizeof...(lhs_t), len2 = sizeof...(rhs_t) };
  typedef tuple<lhs_t..., rhs_t...> type_t;

  MGPU_HOST_DEVICE static type_t cat(tuple<lhs_t...> lhs, tuple<rhs_t...> rhs) {
    type_t result;
    tuple_iterate_assign_t<0, 0, len1>::assign(assign_t(), lhs, result);
    tuple_iterate_assign_t<0, len1, len2>::assign(assign_t(), rhs, result);
    return result;
  }
};

template<typename... lhs_t, typename... rhs_t, typename... others_t>
struct tuple_cat_t<tuple<lhs_t...>, tuple<rhs_t...>, others_t...> {

  typedef typename tuple_cat_t<tuple<rhs_t...>, others_t...>::type_t t2_t;
  typedef typename tuple_cat_t<tuple<lhs_t...>, t2_t>::type_t type_t;

  MGPU_HOST_DEVICE static type_t cat(tuple<lhs_t...> lhs, tuple<rhs_t...> rhs,
    others_t... others) {

    return tuple_cat_t<tuple<lhs_t...>, t2_t>::cat(
      lhs,
      tuple_cat_t<tuple<rhs_t...>, others_t...>::cat(rhs, others...)
    );
  }
};

} // namespace detail

template<typename... args_t>
MGPU_HOST_DEVICE typename detail::tuple_cat_t<args_t...>::type_t
tuple_cat(args_t... args) {
  return detail::tuple_cat_t<args_t...>::cat(args...);
}

////////////////////////////////////////////////////////////////////////////////
// tuple_expand. Calls a function and passes the provided tuple as unpacked
// arguments.

template<typename func_t, typename... args_t, int... seq_i>
MGPU_HOST_DEVICE void tuple_expand(func_t f, tuple<args_t...> tpl, 
  seq_t<seq_i...>) {
  f(get<seq_i>(tpl)...);
}

template<typename func_t, typename... args_t>
MGPU_HOST_DEVICE void tuple_expand(func_t f, tuple<args_t...> tpl) {
  tuple_expand(f, tpl, typename genseq_t<sizeof...(args_t)>::type_t());
}


////////////////////////////////////////////////////////////////////////////////
// Convert a tuple<> of pointer/iterator types to a tuple of the equivalent 
// value types by issuing a load.

template<typename... args_t>
using tuple_iterator_value_t = tuple_convert_t<std::iterator_traits, args_t...>;

namespace detail {

struct load_tuple_op_t {
  template<typename input_t, typename output_t>
  MGPU_HOST_DEVICE void operator()(input_t input, output_t& output, 
    int index) {
    output = input[index];
  }
};

} // namespace detail

template<typename... args_t>
MGPU_HOST_DEVICE typename tuple_iterator_value_t<tuple<args_t...> >::type_t
load_tuple(tuple<args_t...> it, int index) {
  typedef typename tuple_iterator_value_t<tuple<args_t...> >::type_t value_t;
  value_t value;
  tuple_iterate_assign_t<0, 0, sizeof...(args_t)>::assign(
    detail::load_tuple_op_t(), it, value, index);
  return value;
}

////////////////////////////////////////////////////////////////////////////////
// Convert a tuple<> of mixed types to a tuple<> of the same mixed types but
// with restricted pointers.

template<typename arg_t>
struct restrict_t {
  typedef typename add_restrict<arg_t>::type value_type;
};

template<typename... args_t>
using tuple_restricted_t = tuple_convert_t<restrict_t, args_t...>;

namespace detail {

struct restricted_op_t {
  template<typename input_t>
  MGPU_HOST_DEVICE void operator()(input_t input, 
    typename add_restrict<input_t>::type& output) {
    output = (typename add_restrict<input_t>::type)input;
  }  
};

} // namespace detail

template<typename... args_t>
MGPU_HOST_DEVICE typename tuple_restricted_t<tuple<args_t...> >::type_t
restrict_tuple(tuple<args_t...> unrestricted) {
  typedef typename tuple_restricted_t<tuple<args_t...> >::type_t value_t;
  value_t restricted;
  tuple_iterate_assign_t<0, 0, sizeof...(args_t)>::assign(
    detail::restricted_op_t(), unrestricted, restricted);
  return restricted;
}

////////////////////////////////////////////////////////////////////////////////
// Provide overloaded operators for tuple<>. Currently only operator+,
// operator- and operator*.

namespace detail {

template<int i, typename tpl_t, template<class> class op_t, 
  int size = tuple_size<tpl_t>::value>
struct tuple_operator_t {
  MGPU_HOST_DEVICE static void apply(tpl_t& a, tpl_t b) {
    op_t<typename tuple_element<i, tpl_t>::type> op;
    get<i>(a) = op(get<i>(a), get<i>(b));
    tuple_operator_t<i + 1, tpl_t, op_t, size>::apply(a, b);
  }
};
template<typename tpl_t, template<class> class op_t, int size>
struct tuple_operator_t<size, tpl_t, op_t, size> {
  MGPU_HOST_DEVICE static void apply(tpl_t& a, tpl_t b) { }
};

} // namespace detail

template<template<class> class op_t, typename tpl_t>
MGPU_HOST_DEVICE tpl_t tuple_reduce(tpl_t a, tpl_t b) {
  detail::tuple_operator_t<0, tpl_t, op_t>::apply(a, b);
  return a;
}

template<typename... args_t>
MGPU_HOST_DEVICE tuple<args_t...> 
operator+(tuple<args_t...> a, tuple<args_t...> b) {
  return tuple_reduce<plus_t>(a, b);
}
template<typename... args_t>
MGPU_HOST_DEVICE tuple<args_t...> 
operator-(tuple<args_t...> a, tuple<args_t...> b) {
  return tuple_reduce<minus_t>(a, b);
}
template<typename... args_t>
MGPU_HOST_DEVICE tuple<args_t...> 
operator*(tuple<args_t...> a, tuple<args_t...> b) {
  return tuple_reduce<multiplies_t>(a, b);
}

////////////////////////////////////////////////////////////////////////////////
// Tuple comparison operators.
/*
template<typename... args_t>
MGPU_HOST_DEVICE bool operator<(tuple<args_t...> a, tuple<args_t...> b) {
  if(get<0>(a) < get<0>(b)) return true;
  if(get<0>(b) < get<0>(a)) return false;
  return a.inner() < b.inner();
}
template<typename... args_t>
MGPU_HOST_DEVICE bool operator<=(tuple<args_t...> a, tuple<args_t...> b) {
  return !(b < a);
}
template<typename... args_t>
MGPU_HOST_DEVICE bool operator>(tuple<args_t...> a, tuple<args_t...> b) {
  return b < a;
}
template<typename... args_t>
MGPU_HOST_DEVICE bool operator>=(tuple<args_t...> a, tuple<args_t...> b) {
  return !(a < b);
}
template<typename... args_t>
MGPU_HOST_DEVICE bool operator==(tuple<args_t...> a, tuple<args_t...> b) {
  return (get<0>(a) == get<0>(b)) && (a.inner() == b.inner());
}
template<typename... args_t>
MGPU_HOST_DEVICE bool operator!=(tuple<args_t...> a, tuple<args_t...> b) {
  return !(a == b);
}

MGPU_HOST_DEVICE bool operator<(tuple<> a, tuple<> b) {
  return false;
}
MGPU_HOST_DEVICE bool operator<=(tuple<> a, tuple<> b) {
  return true;
}
MGPU_HOST_DEVICE bool operator>(tuple<> a, tuple<> b) {
  return false;
}
MGPU_HOST_DEVICE bool operator>=(tuple<> a, tuple<> b) {
  return true;
}
MGPU_HOST_DEVICE bool operator==(tuple<> a, tuple<> b) {
  return true;
}
MGPU_HOST_DEVICE bool operator!=(tuple<> a, tuple<> b) {
  return false;
}
*/
////////////////////////////////////////////////////////////////////////////////

// tuple_union_size_t.
// returns the max of the sizeof each element in the tuple.
template<typename... args_t>
struct tuple_union_size_t;

template<typename arg_t, typename... args_t>
struct tuple_union_size_t<tuple<arg_t, args_t...> > {
  enum { 
    inner_size = tuple_union_size_t<tuple<args_t...> >::value,
    outer_size = sizeof(arg_t),
    value = outer_size > inner_size ? outer_size : inner_size
  };
};
template<typename arg_t>
struct tuple_union_size_t<tuple<arg_t> > {
  enum { value = sizeof(arg_t) };
};
template<>
struct tuple_union_size_t<tuple<> > {
  enum { value = 0 };
};

END_MGPU_NAMESPACE
