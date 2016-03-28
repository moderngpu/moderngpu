// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once
#include "search.hxx"

BEGIN_MGPU_NAMESPACE

template<typename launch_arg_t = empty_t,
  typename input_it, typename indices_it, typename output_it>
void bulk_remove(input_it input, int count, indices_it indices, 
  int num_indices, output_it output, context_t& context) {

  typedef typename conditional_typedef_t<launch_arg_t, 
    launch_box_t<
      arch_20_cta<128, 15>,
      arch_35_cta<128, 11>,
      arch_52_cta<128, 15>
    >
  >::type_t launch_t;

  typedef typename std::iterator_traits<input_it>::value_type type_t;

  // Map the removal indices into tiles.
  mem_t<int> partitions = binary_search_partitions<bounds_lower>(indices, 
    count, num_indices, launch_t::nv(context), context);
  const int* p_data = partitions.data();

  auto k = [=]MGPU_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t; 
    enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };
    
    __shared__ union {
      int indices[nv + 1];
    } shared;

    range_t tile = get_tile(cta, nv, count);

    // Search the begin and end iterators to load.
    int begin = p_data[cta];
    int end = p_data[cta + 1]; 
    int b_count = end - begin;

    int* a_shared = shared.indices;
    int* b_shared = shared.indices + tile.count() - b_count;

    // Store the indices to shared memory.
    // TODO: MODIFY MEM_TO_SHARED TO UNCONDITIONALLY WRITE TO FULL SMEM.
    mem_to_shared<nt, vt>(indices + begin, tid, b_count, b_shared, false);

    // Binary search into the remove array to prepare a range for the thread.
    merge_range_t range = {
      // a range
      vt * tid, 
      tile.count(), 
      
      // b range
      binary_search<bounds_lower>(b_shared, b_count, 
        tile.begin + vt * tid, less_t<int>()),
      b_count
    };

    // Emit all values that aren't removed.
    iterate<vt>([&](int i) {
      bool p = range.a_valid() && (!range.b_valid() || 
        tile.begin + range.a_begin < b_shared[range.b_begin]);
      if(p)
        a_shared[range.a_begin - range.b_begin] = tile.begin + range.a_begin;
      else 
        ++range.b_begin;
      ++range.a_begin;
    });
    __syncthreads();

    // Pull the gather indices out of shared memory in strided order.
    array_t<int, vt> gather = shared_to_reg_strided<nt, vt>(
      shared.indices, tid);

    // Gather the elements from input.
    int num_move = tile.count() - b_count;
    array_t<type_t, vt> values;
    strided_iterate<nt, vt, 0>([&](int i, int j) {
      values[i] = input[gather[i]];
    }, tid, num_move);

    // Stream to output.
    reg_to_mem_strided<nt, vt>(values, tid, num_move, 
      output + tile.begin - begin);
  };
  cta_transform<launch_t>(k, count, context);
}

END_MGPU_NAMESPACE
