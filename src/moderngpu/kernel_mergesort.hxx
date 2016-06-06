// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once

#include "transform.hxx"
#include "kernel_merge.hxx"
#include "cta_mergesort.hxx"
#include "intrinsics.hxx"

BEGIN_MGPU_NAMESPACE
  
template<typename keys_it, typename comp_t>
mem_t<int> merge_sort_partitions(keys_it keys, int count, int coop, 
  int spacing, comp_t comp, context_t& context) {

  int num_partitions = div_up(count, spacing) + 1;
  auto k = [=]MGPU_DEVICE(int index) {
    merge_range_t range = compute_mergesort_range(count, index, coop, spacing);
    int diag = min(spacing * index, count) - range.a_begin;
    return merge_path<bounds_lower>(keys + range.a_begin, range.a_count(), 
      keys + range.b_begin, range.b_count(), diag, comp);
  };

  return fill_function<int>(k, num_partitions, context);
}

// Key-value mergesort.
template<typename launch_arg_t = empty_t, typename key_t, typename val_t,
  typename comp_t>
void mergesort(key_t* keys_input, val_t* vals_input, int count,
  comp_t comp, context_t& context) {

  enum { has_values = !std::is_same<val_t, empty_t>::value };

  typedef typename conditional_typedef_t<launch_arg_t, 
    launch_box_t<
      arch_20_cta<128, 17>,
      arch_35_cta<128, 11>,
      arch_52_cta<128, 15>
    >
  >::type_t launch_t;

  int nv = launch_t::nv(context);
  int num_ctas = div_up(count, nv);
  int num_passes = find_log2(num_ctas, true);

  mem_t<key_t> keys_temp(num_passes ? count : 0, context);
  key_t* keys_output = keys_temp.data();

  mem_t<val_t> vals_temp(has_values && num_passes ? count : 0, context);
  val_t* vals_output = vals_temp.data();

  key_t* keys_blocksort = (1 & num_passes) ? keys_output : keys_input;
  val_t* vals_blocksort = (1 & num_passes) ? vals_output : vals_input;

  auto k = [=] MGPU_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };
    typedef cta_sort_t<nt, vt, key_t, val_t> sort_t;

    __shared__ union {
      typename sort_t::storage_t sort;
      key_t keys[nv];
      val_t vals[nv];
    } shared;

    range_t tile = get_tile(cta, nv, count);

    // Load the keys and values.
    kv_array_t<key_t, val_t, vt> unsorted;
    unsorted.keys = mem_to_reg_thread<nt, vt>(keys_input + tile.begin, tid, 
      tile.count(), shared.keys);
    if(has_values)
      unsorted.vals = mem_to_reg_thread<nt, vt>(vals_input + tile.begin, tid,
        tile.count(), shared.vals);

    // Blocksort.
    kv_array_t<key_t, val_t, vt> sorted = sort_t().block_sort(unsorted,
      tid, tile.count(), comp, shared.sort);

    // Store the keys and values.
    reg_to_mem_thread<nt, vt>(sorted.keys, tid, tile.count(), 
      keys_blocksort + tile.begin, shared.keys);
    if(has_values)
      reg_to_mem_thread<nt, vt>(sorted.vals, tid, tile.count(), 
        vals_blocksort + tile.begin, shared.vals);
  };

  cta_transform<launch_t>(k, count, context);

  if(1 & num_passes) {
    std::swap(keys_input, keys_output);
    std::swap(vals_input, vals_output);
  }

  for(int pass = 0; pass < num_passes; ++pass) {
    int coop = 2<< pass;
    mem_t<int> partitions = merge_sort_partitions(keys_input, count, coop,
      nv, comp, context);
    int* mp_data = partitions.data();

    auto k = [=] MGPU_DEVICE(int tid, int cta) {
      typedef typename launch_t::sm_ptx params_t;
      enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };

      __shared__ union {
        key_t keys[nv + 1];
        int indices[nv];
      } shared;

      range_t tile = get_tile(cta, nv, count);

      // Load the range for this CTA and merge the values into register.
      merge_range_t range = compute_mergesort_range(count, cta, coop, nv, 
        mp_data[cta + 0], mp_data[cta + 1]);

      merge_pair_t<key_t, vt> merge = cta_merge_from_mem<bounds_lower, nt, vt>(
        keys_input, keys_input, range, tid, comp, shared.keys);

      // Store merged values back out.
      reg_to_mem_thread<nt>(merge.keys, tid, tile.count(), 
        keys_output + tile.begin, shared.keys);

      if(has_values) {
        // Transpose the indices from thread order to strided order.
        array_t<int, vt> indices = reg_thread_to_strided<nt>(merge.indices,
          tid, shared.indices);

        // Gather the input values and merge into the output values.
        transfer_two_streams_strided<nt>(vals_input + range.a_begin, 
          range.a_count(), vals_input + range.b_begin, range.b_count(),
          indices, tid, vals_output + tile.begin);
      }
    };
    cta_transform<launch_t>(k, count, context);

    std::swap(keys_input, keys_output);
    std::swap(vals_input, vals_output);
  }
}

// Key-only mergesort
template<typename launch_arg_t = empty_t, typename key_t, typename comp_t>
void mergesort(key_t* keys_input, int count, comp_t comp, 
  context_t& context) {

  mergesort<launch_arg_t>(keys_input, (empty_t*)nullptr, count, comp, 
    context);
}

END_MGPU_NAMESPACE
