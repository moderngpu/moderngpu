// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once

#include "cta_reduce.hxx"
#include "cta_scan.hxx"
#include "memory.hxx"
#include "operators.hxx"

BEGIN_MGPU_NAMESPACE

template<scan_type_t scan_type = scan_type_exc, 
  typename launch_arg_t = empty_t, typename input_it, 
  typename output_it, typename op_t, typename reduction_it>
void scan_event(input_it input, int count, output_it output, op_t op, 
  reduction_it reduction, context_t& context, cudaEvent_t event) {

  typedef typename conditional_typedef_t<launch_arg_t, 
    launch_box_t<
      arch_20_cta<128, 11>,
      arch_35_cta<128, 7>,
      arch_52_cta<128, 11>
    >
  >::type_t launch_t;

  typedef typename std::iterator_traits<input_it>::value_type type_t;

  int num_ctas = launch_t::cta_dim(context).num_ctas(count);

  if(num_ctas > 8) {
    mem_t<type_t> partials(num_ctas, context);
    type_t* partials_data = partials.data();

    ////////////////////////////////////////////////////////////////////////////
    // Upsweep phase. Reduce each tile to a scalar and store to partials.

    auto upsweep_k = [=] MGPU_DEVICE(int tid, int cta) {
      typedef typename launch_t::sm_ptx params_t;
      enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };
      typedef cta_reduce_t<nt, type_t> reduce_t;

      __shared__ union {
        typename reduce_t::storage_t reduce;
      } shared;

      // Load the tile's data into register.
      range_t tile = get_tile(cta, nv, count);
      array_t<type_t, vt> x = mem_to_reg_strided<nt, vt>(input + tile.begin,
        tid, tile.count());

      // Reduce the thread's values into a scalar.
      type_t scalar = type_t();
      strided_iterate<nt, vt>([&](int i, int j) {
        scalar = i ? op(scalar, x[i]) : x[0];
      }, tid, tile.count());

      // Reduce across all threads.
      type_t all_reduce = reduce_t().reduce(tid, scalar, shared.reduce, 
        tile.count(), op);

      // Store the final reduction to the partials.
      if(!tid)
        partials_data[cta] = all_reduce;
    };
    cta_transform<launch_t>(upsweep_k, count, context);

    ////////////////////////////////////////////////////////////////////////////
    // Spine phase. Recursively call scan on the CTA partials.

    scan_event<scan_type_exc>(partials_data, num_ctas, partials_data,
      op, reduction, context, event);

    // Record the event. This lets the caller wait on just the reduction 
    // part of the operation. It's useful when writing the reduction to
    // host-side paged-locked memory; the caller can read out the value more
    // quickly to allocate memory and launch the next kernel.
    if(event)
      cudaEventRecord(event, context.stream());

    ////////////////////////////////////////////////////////////////////////////
    // Downsweep phase. Perform an intra-tile scan and add the scan of the 
    // partials as carry-in.

    auto downsweep_k = [=] MGPU_DEVICE(int tid, int cta) {
      typedef typename launch_t::sm_ptx params_t;
      enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };
      typedef cta_scan_t<nt, type_t> scan_t;

      __shared__ union {
        typename scan_t::storage_t scan;
        type_t values[nv];
      } shared;

      // Load a tile to register in thread order.
      range_t tile = get_tile(cta, nv, count);
      array_t<type_t, vt> x = mem_to_reg_thread<nt, vt>(input + tile.begin, 
        tid, tile.count(), shared.values);

      // Scan the array with carry-in from the partials.
      array_t<type_t, vt> y = scan_t().scan(tid, x, shared.scan, 
        partials_data[cta], cta > 0, tile.count(), op, type_t(), 
        scan_type).scan;

      // Store the scanned values to the output.
      reg_to_mem_thread<nt, vt>(y, tid, tile.count(), output + tile.begin, 
        shared.values);    
    };
    cta_transform<launch_t>(downsweep_k, count, context);
  
  } else {

    ////////////////////////////////////////////////////////////////////////////
    // Small input specialization. This is the non-recursive branch.

    typedef launch_params_t<512, 3> spine_params_t;
    auto spine_k = [=] MGPU_DEVICE(int tid, int cta) {
     
      enum { nt = spine_params_t::nt, vt = spine_params_t::vt, nv = nt * vt };
      typedef cta_scan_t<nt, type_t> scan_t;

      __shared__ union {
        typename scan_t::storage_t scan;
        type_t values[nv];
      } shared;

      type_t carry_in = type_t();
      for(int cur = 0; cur < count; cur += nv) {
        // Cooperatively load values into register.
        int count2 = min<int>(count - cur, nv);
        array_t<type_t, vt> x = mem_to_reg_thread<nt, vt>(input + cur, 
          tid, count2, shared.values);

        scan_result_t<type_t, vt> result = scan_t().scan(tid, x, shared.scan,
          carry_in, cur > 0, count2, op, type_t(), scan_type);

        // Store the scanned values back to global memory.
        reg_to_mem_thread<nt, vt>(result.scan, tid, count2, 
          output + cur, shared.values);
        
        // Roll the reduction into carry_in.
        carry_in = result.reduction;
      }

      // Store the carry-out to the reduction pointer. This may be a
      // discard_iterator_t if no reduction is wanted.
      if(!tid)
        *reduction = carry_in;
    };
    cta_launch<spine_params_t>(spine_k, 1, context);

    // Record the event. This lets the caller wait on just the reduction 
    // part of the operation. It's useful when writing the reduction to
    // host-side paged-locked memory; the caller can read out the value more
    // quickly to allocate memory and launch the next kernel.
    if(event)
      cudaEventRecord(event, context.stream());
  }
}

template<scan_type_t scan_type = scan_type_exc, 
  typename launch_arg_t = empty_t, typename input_it, 
  typename output_it, typename op_t, typename reduction_it>
void scan(input_it input, int count, output_it output, op_t op, 
  reduction_it reduction, context_t& context) {
  return scan_event<scan_type, launch_arg_t>(input, count, output, op, 
    reduction, context, 0);
}

template<scan_type_t scan_type = scan_type_exc, 
  typename launch_arg_t = empty_t, 
  typename input_it, typename output_it>
void scan(input_it input, int count, output_it output, context_t& context) {

  typedef typename std::iterator_traits<input_it>::value_type type_t;
  scan<scan_type, launch_arg_t>(input, count, output, plus_t<type_t>(),
    discard_iterator_t<type_t>(), context);
}

template<typename type_t, scan_type_t scan_type = scan_type_exc, 
  typename launch_arg_t = empty_t, typename func_t, typename output_it,
  typename op_t, typename reduction_it>
void transform_scan_event(func_t f, int count, output_it output, op_t op,
  reduction_it reduction, context_t& context, cudaEvent_t event) {

  scan_event<scan_type, launch_arg_t>(make_load_iterator<type_t>(f),
    count, output, op, reduction, context, event);
}

template<typename type_t, scan_type_t scan_type = scan_type_exc, 
  typename launch_arg_t = empty_t, typename func_t, typename output_it,
  typename op_t, typename reduction_it>
void transform_scan(func_t f, int count, output_it output, op_t op,
  reduction_it reduction, context_t& context) {

  transform_scan_event<type_t, scan_type, launch_arg_t>(f, count, output, op,
    reduction, context, 0);
}

END_MGPU_NAMESPACE
