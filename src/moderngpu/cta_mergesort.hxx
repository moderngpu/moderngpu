// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once

#include "cta_merge.hxx"
#include "sort_networks.hxx"

BEGIN_MGPU_NAMESPACE

MGPU_HOST_DEVICE int out_of_range_flags(int first, int vt, int count) {
  int out_of_range = min(vt, first + vt - count);
  int head_flags = 0;
  if(out_of_range > 0) {
    const int mask = (1<< vt) - 1;
    head_flags = mask & (~mask>> out_of_range);
  }
  return head_flags;
}

MGPU_HOST_DEVICE merge_range_t compute_mergesort_frame(int partition,
  int coop, int spacing) {

  int size = spacing * (coop / 2);
  int start = ~(coop - 1) & partition;
  int a_begin = spacing * start;
  int b_begin = spacing * start + size;

  return merge_range_t {
    a_begin,
    a_begin + size,
    b_begin,
    b_begin + size
  };
}

MGPU_HOST_DEVICE merge_range_t compute_mergesort_range(int count, 
  int partition, int coop, int spacing) {

  merge_range_t frame = compute_mergesort_frame(partition, coop, spacing);

  return merge_range_t {
    frame.a_begin,
    min(count, frame.a_end),
    min(count, frame.b_begin),
    min(count, frame.b_end)
  };
}

MGPU_HOST_DEVICE merge_range_t compute_mergesort_range(int count, 
  int partition, int coop, int spacing, int mp0, int mp1) {

  merge_range_t range = compute_mergesort_range(count, partition, 
    coop, spacing);

  // Locate the diagonal from the start of the A sublist.
  int diag = spacing * partition - range.a_begin;

  // The end partition of the last cta for each merge operation is computed
  // and stored as the begin partition for the subsequent merge. i.e. it is
  // the same partition but in the wrong coordinate system, so its 0 when it
  // should be listSize. Correct that by checking if this is the last cta
  // in this merge operation.
  if(coop - 1 != ((coop - 1) & partition)) {
    range.a_end = range.a_begin + mp1;
    range.b_end = min(count, range.b_begin + diag + spacing - mp1);
  }

  range.a_begin = range.a_begin + mp0;
  range.b_begin = min(count, range.b_begin + diag - mp0);

  return range;
}

template<int nt, int vt, typename key_t, typename val_t>
struct cta_sort_t {
  enum { 
    has_values = !std::is_same<val_t, empty_t>::value,
    num_passes = s_log2(nt)
  };

  union storage_t {
    key_t keys[nt * vt + 1];
    val_t vals[nt * vt];
  };

  static_assert(is_pow2(nt), "cta_sort_t requires pow2 number of threads");

  template<typename comp_t>
  MGPU_DEVICE kv_array_t<key_t, val_t, vt> 
  merge_pass(kv_array_t<key_t, val_t, vt> x, int tid, int count, 
    int pass, comp_t comp, storage_t& storage) const {

    // Divide the CTA's keys into lists.
    int coop = 2<< pass;
    merge_range_t range = compute_mergesort_range(count, tid, coop, vt);
    int diag = vt * tid - range.a_begin;

    // Store the keys into shared memory for searching.
    reg_to_shared_thread<nt, vt>(x.keys, tid, storage.keys);
    
    // Search for the merge path for this thread within its list.
    int mp = merge_path<bounds_lower>(storage.keys, range, diag, comp);

    // Run a serial merge and return.
    merge_pair_t<key_t, vt> merge = serial_merge<bounds_lower, vt>(
      storage.keys, range.partition(mp, diag), comp);
    x.keys = merge.keys;

    if(has_values) {
      // Reorder values through shared memory.
      reg_to_shared_thread<nt, vt>(x.vals, tid, storage.vals);
      x.vals = shared_gather<nt, vt>(storage.vals, merge.indices);
    }

    return x;
  }

  template<typename comp_t>
  MGPU_DEVICE kv_array_t<key_t, val_t, vt> 
  block_sort(kv_array_t<key_t, val_t, vt> x, int tid, int count,
    comp_t comp, storage_t& storage) const {

    // Sort the inputs within each thread. If any threads have fewer than
    // vt items, use the segmented sort network to prevent out-of-range
    // elements from contaminating the sort.
    if(count < nt * vt) {
      int head_flags = out_of_range_flags(vt * tid, vt, count);
      x = odd_even_sort(x, comp, head_flags);
    } else
      x = odd_even_sort(x, comp);

    // Merge threads starting with a pair until all values are merged.
    for(int pass = 0; pass < num_passes; ++pass)
      x = merge_pass(x, tid, count, pass, comp, storage);
    
    return x;
  }
};


END_MGPU_NAMESPACE
