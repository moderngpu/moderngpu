// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once
#include "cta_merge.hxx"
#include "search.hxx"

BEGIN_MGPU_NAMESPACE

// Key-value merge.
template<typename launch_arg_t = empty_t,
  typename a_keys_it, typename a_vals_it, 
  typename b_keys_it, typename b_vals_it,
  typename c_keys_it, typename c_vals_it, 
  typename comp_t>
void merge(a_keys_it a_keys, a_vals_it a_vals, int a_count, 
  b_keys_it b_keys, b_vals_it b_vals, int b_count,
  c_keys_it c_keys, c_vals_it c_vals, comp_t comp, context_t& context) {

  typedef typename conditional_typedef_t<launch_arg_t, 
    launch_box_t<
      arch_20_cta<128, 15>,
      arch_35_cta<128, 11>,
      arch_52_cta<128, 15>
    >
  >::type_t launch_t;

  typedef typename std::iterator_traits<a_keys_it>::value_type type_t;
  typedef typename std::iterator_traits<a_vals_it>::value_type val_t;
  enum { has_values = !std::is_same<val_t, empty_t>::value };

  mem_t<int> partitions = merge_path_partitions<bounds_lower>(a_keys, a_count, 
    b_keys, b_count, launch_t::nv(context), comp, context);
  int* mp_data = partitions.data();

  auto k = [=] MGPU_DEVICE (int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };

    __shared__ union {
      type_t keys[nv + 1];
      int indices[nv];
    } shared;

    // Load the range for this CTA and merge the values into register.
    int mp0 = mp_data[cta + 0];
    int mp1 = mp_data[cta + 1];
    merge_range_t range = compute_merge_range(a_count, b_count, cta, nv, 
      mp0, mp1);

    merge_pair_t<type_t, vt> merge = cta_merge_from_mem<bounds_lower, nt, vt>(
      a_keys, b_keys, range, tid, comp, shared.keys);

    int dest_offset = nv * cta;
    reg_to_mem_thread<nt>(merge.keys, tid, range.total(), c_keys + dest_offset,
      shared.keys);

    if(has_values) {
      // Transpose the indices from thread order to strided order.
      array_t<int, vt> indices = reg_thread_to_strided<nt>(merge.indices, tid, 
        shared.indices);

      // Gather the input values and merge into the output values.
      transfer_two_streams_strided<nt>(a_vals + range.a_begin, range.a_count(),
        b_vals + range.b_begin, range.b_count(), indices, tid, 
        c_vals + dest_offset);
    }
  };
  cta_transform<launch_t>(k, a_count + b_count, context);
}

// Key-only merge.
template<typename launch_t = empty_t,
  typename a_keys_it, typename b_keys_it, typename c_keys_it,
  typename comp_t>
void merge(a_keys_it a_keys, int a_count, b_keys_it b_keys, int b_count,
  c_keys_it c_keys, comp_t comp, context_t& context) {

  merge<launch_t>(a_keys, (const empty_t*)nullptr, a_count, b_keys, 
    (const empty_t*)nullptr, b_count, c_keys, (empty_t*)nullptr, comp,
    context);
}

END_MGPU_NAMESPACE
