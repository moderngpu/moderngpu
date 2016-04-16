// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once

#include "cta_reduce.hxx"
#include "memory.hxx"
#include "transform.hxx"
#include "operators.hxx"

BEGIN_MGPU_NAMESPACE

template<typename launch_arg_t = empty_t, typename input_it, 
  typename output_it, typename op_t>
void reduce(input_it input, int count, output_it reduction, op_t op, 
  context_t& context) {

  typedef typename conditional_typedef_t<launch_arg_t, 
    launch_params_t<128, 8>
  >::type_t launch_t;

  typedef typename std::iterator_traits<input_it>::value_type type_t;

  int num_ctas = launch_t::cta_dim(context).num_ctas(count);
  mem_t<type_t> partials(num_ctas, context);
  type_t* partials_data = partials.data();

  auto k = [=] MGPU_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };
    typedef cta_reduce_t<nt, type_t> reduce_t;
    __shared__ typename reduce_t::storage_t shared_reduce;

    // Load the data for the first tile for each cta.
    range_t tile = get_tile(cta, nv, count);
    array_t<type_t, vt> x = mem_to_reg_strided<nt, vt>(input + tile.begin, 
      tid, tile.count());

    // Reduce the multiple values per thread into a scalar.
    type_t scalar;
    strided_iterate<nt, vt>([&](int i, int j) {
      scalar = i ? op(scalar, x[i]) : x[0];
    }, tid, tile.count());

    // Reduce to a scalar per CTA.
    scalar = reduce_t().reduce(tid, scalar, shared_reduce, 
      min(tile.count(), (int)nt), op, false);

    if(!tid) {
      if(1 == num_ctas) *reduction = scalar;
      else partials_data[cta] = scalar;
    }
  };
  cta_launch<launch_t>(k, num_ctas, context);

  // Recursively call reduce until there's just one scalar.
  if(num_ctas > 1)
    reduce<launch_params_t<512, 4> >(partials_data, num_ctas, reduction, op, 
      context);
}

template<typename launch_arg_t = empty_t, typename func_t, 
  typename output_it, typename op_t>
void transform_reduce(func_t f, int count, output_it reduction, op_t op, 
  context_t& context) {

  typedef typename std::iterator_traits<output_it>::value_type type_t;
  reduce<launch_arg_t>(make_load_iterator<type_t>(f), count, reduction, op, 
    context);
}

END_MGPU_NAMESPACE
