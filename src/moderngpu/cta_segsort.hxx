// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once

#include "cta_mergesort.hxx"

BEGIN_MGPU_NAMESPACE

template<typename keys_it, typename comp_t>
MGPU_HOST_DEVICE int segmented_merge_path(keys_it keys, merge_range_t range,
  range_t active, int diag, comp_t comp) {

  // Consider a rectangle defined by range.
  // Now consider a sub-rectangle at the top-right corner defined by
  // active. We want to run the merge path only within this corner part.

  // If the cross-diagonal does not intersect our corner, return immediately.
  if(range.a_begin + diag <= active.begin) return diag;
  if(range.a_begin + diag >= active.end) return range.a_count();

  // Call merge_path on the corner domain.
  active.begin = max(active.begin, range.a_begin);
  active.end = min(active.end, range.b_end);

  merge_range_t active_range = { 
    active.begin, range.a_end, 
    range.b_begin, active.end 
  };

  int active_offset = active.begin - range.a_begin;
  int p = merge_path<bounds_lower>(keys, active_range, 
    diag - active_offset, comp);

  return p + active_offset;
}

template<int vt, typename type_t, typename comp_t>
MGPU_DEVICE merge_pair_t<type_t, vt> segmented_serial_merge(
  const type_t* keys_shared, merge_range_t range, range_t active, 
  comp_t comp, bool sync = true) {

  range.b_end = min(active.end, range.b_end);

  type_t a_key = keys_shared[range.a_begin];
  type_t b_key = keys_shared[range.b_begin];

  merge_pair_t<type_t, vt> merge_pair;
  iterate<vt>([&](int i) {
    bool p;
    if(range.a_begin >= range.a_end) 
      // If A has run out of inputs, emit B.
      p = false;
    else if(range.b_begin >= range.b_end || range.a_begin < active.begin)
      // B has hit the end of the middle segment.
      // Emit A if A has inputs remaining in the middle segment.
      p = true;
    else 
      // Emit the smaller element in the middle segment.
      p = !comp(b_key, a_key);

    int index = p ? range.a_begin : range.b_begin;
    merge_pair.keys[i] = p ? a_key : b_key;
    merge_pair.indices[i] = index;

    type_t c_key = keys_shared[++index];
    if(p) a_key = c_key, range.a_begin = index;
    else b_key = c_key, range.b_begin = index;
  });
  
  if(sync) __syncthreads();
  return merge_pair;
}

template<int nt, int vt>
struct cta_load_head_flags {
  enum { 
    nv = nt * vt, 

    // Store each flag in a byte; there are 4 bytes in a word, and threads
    // cooperatively reset these.
    words_per_thread = div_up(vt, 32 / 8)
  };

  union storage_t {
    char flags[nv];
    int words[nt * words_per_thread];
  };

  template<typename seg_it>
  MGPU_DEVICE int load(seg_it segments, const int* partitions_global,
    int tid, int cta, int count, storage_t& storage) {

    int mp0 = partitions_global[0];
    int mp1 = partitions_global[1];
    int gid = nv * cta;
    count -= gid;

    // Set the head flags for out-of-range keys.
    int head_flags = out_of_range_flags(vt * tid, vt, count);

    if(mp1 > mp0) {
      // Clear the flag bytes, then loop through the indices and poke in
      // flag bytes.
      iterate<words_per_thread>([&](int i) { 
        storage.words[nt * i + tid] = 0;
      });
      __syncthreads();

      for(int index = mp0 + tid; index < mp1; index += nt)
        storage.flags[segments[index] - gid] = 1;
      __syncthreads();

      // Combine all the head flags for this thread.
      int first = vt * tid;
      int offset = first / 4;
      int prev = storage.words[offset];
      int mask = 0x3210 + 0x1111 * (3 & first);
      iterate<words_per_thread>([&](int i) {
        int next = storage.words[offset + 1 + i];
        int x = prmt(prev, next, mask);
        prev = next;

        // Set the head flag bits.
        if(0x00000001 & x) head_flags |= 1<< (4 * i + 0);
        if(0x00000100 & x) head_flags |= 1<< (4 * i + 1);
        if(0x00010000 & x) head_flags |= 1<< (4 * i + 2);
        if(0x01000000 & x) head_flags |= 1<< (4 * i + 3);
      });
      head_flags &= (1<< vt) - 1;
      __syncthreads();
    }

    return head_flags;
  }
};

template<int nt, int vt, typename key_t, typename val_t>
struct cta_segsort_t {
  enum { 
    nv = nt * vt,
    has_values = !std::is_same<val_t, empty_t>::value,
    num_passes = s_log2(nt)
  };

  struct storage_t {
    union {
      key_t keys[nt * vt];
      val_t vals[nt * vt];
    };
    int ranges[nt];
  };

  static_assert(is_pow2(nt), "cta_segsort_t requires pow2 number of threads");

  template<typename comp_t>
  MGPU_DEVICE kv_array_t<key_t, val_t, vt>
  merge_pass(kv_array_t<key_t, val_t, vt> x, int tid, int count, 
    int pass, range_t& active, comp_t comp, storage_t& storage) const {

    int coop = 2<< pass;
    merge_range_t range = compute_mergesort_range(count, tid, coop, vt);

    int list = tid>> pass;

    int list_parity = 1 & list;
    int diag = vt * tid - range.a_begin;

    // Fetch the active range for the list this thread's list is merging with.
    int sibling_range = storage.ranges[1 ^ list];
    range_t sibling { 0x0000ffff & sibling_range, sibling_range>> 16 };

    // This pass does a segmented merge on ranges list and 1 ^ list.
    // ~1 & list is the left list and 1 | list is the right list.
    // We find the inner segments for merging, then update the active
    // range to the outer segments for the next pass.
    range_t left = list_parity ? sibling : active;
    range_t right = list_parity ? active : sibling;
    range_t inner = { left.end, right.begin };
    active.begin = min(left.begin, right.begin);
    active.end = max(left.end, right.end);

    // Store the data from thread order into shared memory.
    reg_to_shared_thread<nt, vt>(x.keys, tid, storage.keys);

    int mp = segmented_merge_path(storage.keys, range, inner, diag, comp);

    // Run a segmented serial merge.
    merge_pair_t<key_t, vt> merge = segmented_serial_merge<vt>(storage.keys,
      range.partition(mp, diag), inner, comp);

    // Pack and store the outer range to shared memory.
    storage.ranges[list>> 1] = (int)bfi(active.end, active.begin, 16, 16);
    if(!has_values) __syncthreads();

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
    int head_flags, range_t& active, comp_t comp, storage_t& storage) const {

    // Sort the inputs within each thread.
    x = odd_even_sort(x, comp, head_flags);

    // Record the first and last occurrences of head flags in this segment.
    active.begin = head_flags ? (vt * tid - 1 + ffs(head_flags)) : nv;
    active.end = head_flags ? (vt * tid + 31 - clz(head_flags)) : -1;
    storage.ranges[tid] = bfi(active.end, active.begin, 16, 16);
    __syncthreads();

    // Merge threads starting with a pair until all values are merged.
    for(int pass = 0; pass < num_passes; ++pass)
      x = merge_pass(x, tid, count, pass, active, comp, storage);
    
    return x;
  }
};

END_MGPU_NAMESPACE
