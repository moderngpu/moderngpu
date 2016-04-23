// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once

#include "kernel_scan.hxx"

BEGIN_MGPU_NAMESPACE

template<typename launch_arg_t>
struct stream_compact_t {

  typedef typename conditional_typedef_t<launch_arg_t, 
    launch_box_t<
      arch_20_cta<128, 11>,
      arch_35_cta<128,  7>,
      arch_52_cta<128, 11>
    >
  >::type_t launch_t;

  cta_dim_t cta_dim;
  int num_ctas;
  int count;
  context_t& context;

  mem_t<short> bits;
  mem_t<int> cta_offsets;

public:
  stream_compact_t(int count_, context_t& context_) : context(context_) {
    count = count_;
    cta_dim = launch_t::cta_dim(context);
    num_ctas = cta_dim.num_ctas(count);

    bits = mem_t<short>(num_ctas * cta_dim.nt, context);
    cta_offsets = mem_t<int>(num_ctas, context);
  }

  // upsweep of stream compaction. 
  // func_t implements bool operator(int index);
  // The return value is flag for indicating that we want to *keep* the data
  // in the compacted stream.
  template<typename func_t>
  int upsweep(func_t f) {
    short* bits_data = bits.data();
    int* cta_offsets_data = cta_offsets.data();
    int count = this->count;

    auto upsweep_k = [=]MGPU_DEVICE(int tid, int cta) {
      typedef typename launch_t::sm_ptx params_t;
      enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };
      typedef cta_reduce_t<nt, int> reduce_t;
      static_assert(vt <= 16, "mgpu::stream_compact_vt must be <= 16.");

      __shared__ union {
        typename reduce_t::storage_t reduce;
      } shared;

      range_t tile = get_tile(cta, nv, count);
      short stream_bits = 0;
      strided_iterate<nt, vt>([&](int i, int j) {
        int index = tile.begin + j;
        bool stream = f(index);
        if(stream) stream_bits |= 1<< i;
      }, tid, tile.count());

      // Reduce the values and store to global memory.
      int total_stream = reduce_t().reduce(tid, popc(stream_bits), 
        shared.reduce, nt, plus_t<int>(), false);

      bits_data[nt * cta + tid] = stream_bits;
      if(!tid) cta_offsets_data[cta] = total_stream;
    };
    cta_launch<launch_t>(upsweep_k, num_ctas, context);

    // Scan reductions.
    mem_t<int> counts_host(1, context, memory_space_host);
    scan_event(cta_offsets_data, num_ctas, cta_offsets_data, 
      plus_t<int>(), counts_host.data(), context, context.event());
    cudaEventSynchronize(context.event());

    // Return the total number of elements to stream.
    int stream_total = counts_host.data()[0];
    return stream_total;
  }

  // downsweep of stream compaction.
  // func_t implements void operator(int dest_index, int source_index).
  template<typename func_t>
  void downsweep(func_t f) {
    const short* bits_data = bits.data();
    const int* cta_offsets_data = cta_offsets.data();

    auto downsweep_k = [=]MGPU_DEVICE(int tid, int cta) {
      typedef typename launch_t::sm_ptx params_t;
      enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };
      typedef cta_scan_t<nt, int> scan_t;
      __shared__ union {
        typename scan_t::storage_t scan;
        short indices[nv];
      } shared;

      short stream_bits = bits_data[nt * cta + tid];
      int cta_offset = cta_offsets_data[cta];

      // For each set stream_bits bit, set shared.indices to 1.
      iterate<vt>([&](int i) {
        shared.indices[nt * i + tid] = 0 != ((1<< i) & stream_bits);
      });
      __syncthreads();

      // Load out the values and scan. Compact into shared.indices the 
      // CTA-local indices of each streaming work-item.
      array_t<short, vt> flags = shared_to_reg_thread<nt, vt>(
        shared.indices, tid);
      scan_result_t<int> scan = scan_t().scan(tid, reduce(flags), 
        shared.scan);
      iterate<vt>([&](int i) {
        if(flags[i]) shared.indices[scan.scan++] = (short)(vt * tid + i);
      });
      __syncthreads();

      // Call the user-supplied callback with destination and source indices.
      for(int i = tid; i < scan.reduction; i += nt) {
        int source_index = nv * cta + shared.indices[i];
        int dest_index = cta_offset + i;
        f(dest_index, source_index);
      }
      __syncthreads();
    };
    cta_launch<launch_t>(downsweep_k, num_ctas, context);
  }
};

template<typename launch_arg_t = empty_t>
stream_compact_t<launch_arg_t> 
transform_compact(int count, context_t& context) {
  return stream_compact_t<launch_arg_t>(count, context);
}

END_MGPU_NAMESPACE
