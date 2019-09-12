// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once

#include "meta.hxx"

BEGIN_MGPU_NAMESPACE

template<typename type_t>
using decay_t = typename std::decay<type_t>::type;

/////////////////
// index_sequence

// Improved linear index_sequence from
// http://talesofcpp.fusionfenix.com/post-22/true-story-efficient-packing
template<size_t... int_s>
struct index_sequence { 
  enum { size = sizeof...(int_s) };
};

namespace detail {
template<typename seq_t>
struct _next;

template<size_t... seq_i>
struct _next<index_sequence<seq_i...> > {
  // grow the sequence by one element.
  typedef index_sequence<seq_i..., sizeof...(seq_i)> type;
};

template<size_t count>
struct _make_index_sequence : 
  _next<typename _make_index_sequence<count - 1>::type> { };

template<> struct _make_index_sequence<0> {
  typedef index_sequence<> type;
};
} // namespace detail

template<size_t count>
using make_index_sequence = 
  typename detail::_make_index_sequence<count>::type;

//////////
// var_and

template<bool... args_b>
struct var_and;

template<bool arg_a, bool... args_b> 
struct var_and<arg_a, args_b...> {
  enum { value = arg_a && var_and<args_b...>::value };
};
template<bool arg_a>
struct var_and<arg_a> {
  enum { value = arg_a };
};
template<>
struct var_and<> {
  enum { value = true };
};

//////////
// var_or

template<bool... args_b>
struct var_or;

template<bool arg_a, bool... args_b> 
struct var_or<arg_a, args_b...> {
  enum { value = arg_a || var_or<args_b...>::value };
};
template<bool arg_a>
struct var_or<arg_a> {
  enum { value = arg_a };
};
template<>
struct var_or<> {
  enum { value = false };
};



// Forward declare the tuple.
template<typename... args_t>
struct tuple;

////////////////
// tuple_element

template<size_t i, typename tpl_t> 
struct tuple_element;

template<size_t i, typename arg_t, typename... args_t>
struct tuple_element<i, tuple<arg_t, args_t...> > : 
  tuple_element<i - 1, tuple<args_t...> > { };

template<typename arg_t, typename... args_t>
struct tuple_element<0, tuple<arg_t, args_t...> > {
  typedef arg_t type;
};

template<size_t i, typename tpl_t>
using tuple_element_t = typename tuple_element<i, tpl_t>::type;

/////////////
// tuple_size

template<typename tpl_t>
struct tuple_size;

template<typename... args_t>
struct tuple_size<tuple<args_t...> > {
  enum { value = sizeof...(args_t) };
};


namespace detail {

template<size_t i, typename arg_t, bool is_empty = std::is_empty<arg_t>::value>
struct tuple_leaf {
  arg_t x;

  MGPU_HOST_DEVICE arg_t& get() { return x; }
  MGPU_HOST_DEVICE const arg_t& get() const { return x; }

  tuple_leaf() = default;
  tuple_leaf(const tuple_leaf&) = default;

  template<typename arg2_t,
    typename = typename std::enable_if<
      std::is_constructible<arg_t, arg2_t&&>::value
    >::type
  > MGPU_HOST_DEVICE 
  tuple_leaf(arg2_t&& arg) : x(std::forward<arg2_t>(arg)) { }

  template<typename arg2_t,
    typename = typename std::enable_if<
      std::is_constructible<arg_t, const arg2_t&>::value
    >::type
  > MGPU_HOST_DEVICE  
  tuple_leaf(const arg2_t& arg) : x(arg) { }
};

template<size_t i, typename arg_t>
struct tuple_leaf<i, arg_t, true> : arg_t { 
  arg_t& get() { return *this; }
  const arg_t& get() const { return *this; }

  template<typename arg2_t,
    typename = typename std::enable_if<
      std::is_constructible<arg_t, const arg2_t&>::value
    >::type
  > MGPU_HOST_DEVICE 
  tuple_leaf(const arg2_t& arg) : arg_t(arg) { }
};

template<size_t i, typename... args_t>
struct tuple_impl;

template<size_t i>
struct tuple_impl<i> { };

template<size_t i, typename arg_t, typename... args_t>
struct tuple_impl<i, arg_t, args_t...> :
  tuple_leaf<i, arg_t>,
  tuple_impl<i + 1, args_t...> {

  typedef tuple_leaf<i, arg_t> head_t;
  typedef tuple_impl<i + 1, args_t...> tail_t;

  MGPU_HOST_DEVICE  arg_t& head() { return head_t::get(); }
  MGPU_HOST_DEVICE const arg_t& head() const { return head_t::get(); }

  MGPU_HOST_DEVICE  tail_t& tail() { return *this; }
  MGPU_HOST_DEVICE  const tail_t& tail() const { return *this; }

  // Constructors.
  tuple_impl() = default;
  explicit tuple_impl(const tuple_impl&) = default;

  template<typename... args2_t> MGPU_HOST_DEVICE 
  explicit tuple_impl(const tuple_impl<i, args2_t...>& rhs) :
    head_t(rhs.head()), tail_t(rhs.tail()) { }

  template<typename... args2_t> MGPU_HOST_DEVICE  
  explicit tuple_impl(tuple_impl<i, args2_t...>&& rhs) :
    head_t(std::move(rhs.head())), 
    tail_t(std::move(rhs.tail())) { }

  template<typename arg2_t, typename... args2_t,
    typename = typename std::enable_if<
      sizeof...(args_t) == sizeof...(args2_t) &&
      std::is_constructible<arg_t, arg2_t&&>::value &&
      var_and<std::is_constructible<args_t, args2_t&&>::value...>::value
    >::type
  > MGPU_HOST_DEVICE 
  tuple_impl(arg2_t&& arg, args2_t&&... args) :
    head_t(std::forward<arg2_t>(arg)), 
    tail_t(std::forward<args2_t>(args)...) { }

  template<typename arg2_t, typename... args2_t,
    typename = typename std::enable_if<
      std::is_constructible<arg_t, const arg2_t&>::value &&
      var_and<std::is_constructible<args_t, const args2_t&>::value...>::value
    >::type
  > MGPU_HOST_DEVICE 
  tuple_impl(const arg2_t& arg, const args2_t&... args) :
    head_t(arg), tail_t(args...) { }

  // Assignment
};

template<size_t i, typename arg_t> MGPU_HOST_DEVICE 
tuple_leaf<i, arg_t>& get_leaf(tuple_leaf<i, arg_t>& leaf) {
  return leaf;
}

template<size_t i, typename arg_t> MGPU_HOST_DEVICE 
const tuple_leaf<i, arg_t>& get_leaf(const tuple_leaf<i, arg_t>& leaf) {
  return leaf;
}

} // namespace detail

template<typename... args_t>
struct tuple : detail::tuple_impl<0, args_t...> { 
  typedef detail::tuple_impl<0, args_t...> impl_t;

  tuple() = default;
  tuple(const tuple&) = default;

  template<typename... args2_t,
    typename = typename std::enable_if<
      sizeof...(args2_t) == sizeof...(args_t) &&
      var_and<std::is_constructible<args_t, const args2_t&>::value...>::value
    >::type
  > MGPU_HOST_DEVICE 
  tuple(const tuple<args2_t...>& rhs) : impl_t(rhs) { }
  
  template<typename... args2_t,
    typename = typename std::enable_if<
      sizeof...(args2_t) == sizeof...(args_t) &&
      var_and<std::is_constructible<args_t, args2_t&&>::value...>::value
    >::type
  > MGPU_HOST_DEVICE 
  tuple(args2_t&&... args) : impl_t(std::forward<args2_t>(args)...) { }

  template<typename... args2_t,
    typename = typename std::enable_if<
      sizeof...(args2_t) == sizeof...(args_t) &&
      var_and<std::is_constructible<args_t, const args2_t&>::value...>::value
    >::type
  > MGPU_HOST_DEVICE  
  tuple(const args2_t&... args) : impl_t(args...) { }
} __attribute__((aligned));

namespace detail {

template<size_t i, typename arg_t> MGPU_HOST_DEVICE 
arg_t& _get(tuple_leaf<i, arg_t>& leaf) {
  return leaf.get();
}

template<size_t i, typename arg_t> MGPU_HOST_DEVICE 
const arg_t& _get(const tuple_leaf<i, arg_t>& leaf) {
  return leaf.get();
}

}

template<size_t i, typename... args_t> MGPU_HOST_DEVICE 
tuple_element_t<i, tuple<args_t...> >&
get(tuple<args_t...>& tpl) {
  return detail::_get<i>(tpl);
}

template<size_t i, typename... args_t> MGPU_HOST_DEVICE 
const tuple_element_t<i, tuple<args_t...> >&
get(const tuple<args_t...>& tpl) {
  return detail::_get<i>(tpl);
}

template<size_t i, typename... args_t> MGPU_HOST_DEVICE 
typename std::add_rvalue_reference<
  tuple_element_t<i, tuple<args_t...> >
>::type
get(tuple<args_t...>&& tpl) {
  return std::forward<tuple_element_t<i, tuple<args_t...> >&&>(get<i>(tpl));
}

template<typename... args_t> MGPU_HOST_DEVICE 
tuple<decay_t<args_t>...> make_tuple(args_t&&... args) {
  return tuple<decay_t<args_t>...>(std::forward<args_t>(args)...);
}

template<typename... args_t> MGPU_HOST_DEVICE
tuple<args_t&&...> forward_as_tuple(args_t&&... args) {
  return tuple<args_t&&...>(std::forward<args_t>(args)...);
}

////////////
// tuple_cat

namespace detail {

template<typename tuple_t>
struct _make_tuple {
  typedef typename std::remove_cv<
    typename std::remove_reference<tuple_t>::type
  >::type type;
};

template<typename... tuples_t>
struct _combine_type;

template<typename... args_t>
struct _combine_type<tuple<args_t...> > {
  typedef tuple<args_t...> type;
};

template<typename... args1_t, typename... args2_t, typename... tuples_t>
struct _combine_type<tuple<args1_t...>, tuple<args2_t...>, tuples_t...> {
  typedef typename _combine_type<
    tuple<args1_t..., args2_t...>,
    tuples_t...
  >::type type;
};

template<typename... tpls_t>
struct _tuple_cat_ret {
  typedef typename _combine_type<
    typename _make_tuple<tpls_t>::type...
  >::type type;
};

template<typename tpl1_t, typename seq1_t, typename tpl2_t, typename seq2_t>
struct _tuple_cat;

template<typename tpl1_t, size_t... seq1_i, typename tpl2_t, size_t... seq2_i>
struct _tuple_cat<tpl1_t, index_sequence<seq1_i...>, 
  tpl2_t, index_sequence<seq2_i...> > {

  typedef typename _tuple_cat_ret<tpl1_t, tpl2_t>::type ret_t;

  MGPU_HOST_DEVICE static ret_t cat(tpl1_t&& tpl1, tpl2_t&& tpl2) {
    return make_tuple(
      get<seq1_i>(std::forward<tpl1_t>(tpl1))...,
      get<seq2_i>(std::forward<tpl2_t>(tpl2))...
    );
  }
};

} // namespace detail

template<typename tpl1_t> MGPU_HOST_DEVICE
typename detail::_tuple_cat_ret<tpl1_t>::type
tuple_cat(tpl1_t&& tpl1) {
  return std::forward<tpl1_t>(tpl1);
}

template<typename tpl1_t, typename tpl2_t, typename... tpls_t> MGPU_HOST_DEVICE
typename detail::_tuple_cat_ret<tpl1_t, tpl2_t, tpls_t...>::type
tuple_cat(tpl1_t&& tpl1, tpl2_t&& tpl2, tpls_t&&... tpls) {
  typedef typename detail::_make_tuple<tpl1_t>::type tpl1_stripped;
  typedef typename detail::_make_tuple<tpl2_t>::type tpl2_stripped;

  enum { 
    size1 = tuple_size<tpl1_stripped>::value, 
    size2 = tuple_size<tpl2_stripped>::value
  };

  return tuple_cat(
    detail::_tuple_cat<
      tpl1_t, make_index_sequence<size1>, 
      tpl2_t, make_index_sequence<size2> 
    >::cat(
      std::forward<tpl1_t>(tpl1), 
      std::forward<tpl2_t>(tpl2)
    ), 
    std::forward<tpls_t>(tpls)...
  );
}

///////
// tie

template<typename... args_t>
MGPU_HOST_DEVICE tuple<args_t&...> tie(args_t&... args) {
  return tuple<args_t&...>(args...);
}

END_MGPU_NAMESPACE
