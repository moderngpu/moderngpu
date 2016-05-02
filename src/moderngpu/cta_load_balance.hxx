// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once
#include "cta_merge.hxx"
#include "operators.hxx"
#include "cpp11.hxx"

BEGIN_MGPU_NAMESPACE

struct lbs_placement_t {
  merge_range_t range;    // The merge range of *loaded* values. 
                          // May extend b_range one element in each direction.
  int a_index;            // Starting A index for merge.
  int b_index;            // Starting B index for merge.
};

template<int nt, int vt, typename segments_it>
MGPU_DEVICE lbs_placement_t cta_load_balance_place(int tid, 
  merge_range_t range, int count, segments_it segments, int num_segments,
  int* b_shared) {

  // We want to know the value of the segment ID for the segment starting
  // this tile. Load it by decrementing range.b_begin.
  int load_preceding = 0 < range.b_begin;
  range.b_begin -= load_preceding;

  // Load a trailing member of the segment ID array. This lets us read one past
  // the last member: b_key = b_shared[++b0]. Note the use of prefix increment,
  // which gets the beginning of the next identifier, not the current one.
  if(range.b_end < num_segments && range.a_end < count)
    ++range.b_end;

  int load_count = range.b_count();
  int fill_count = nt * vt + 1 + load_preceding - load_count - range.a_count();

  // Fill the end of the array with dest_count.
  for(int i = tid; i < fill_count; i += nt)
    b_shared[load_count + i] = count;

  // Load the segments descriptors into the front of the indices array.
  // TODO: SUBTRACT OUT A_BEGIN FROM B_BEGIN SO WE CAN DO 32-BIT COMPARISONS!
  for(int i = tid; i < load_count; i += nt)
    b_shared[i] = segments[range.b_begin + i];
  __syncthreads();

  // Run a merge path search to find the start of the serial merge for
  // each thread. If we loaded a preceding value from B, increment the 
  // cross-diagonal so that we don't redundantly process it.
  int diag = vt * tid + load_preceding;
  int mp = merge_path<bounds_upper>(counting_iterator_t<int>(range.a_begin),
    range.a_count(), b_shared, load_count + fill_count, diag, less_t<int>());
  __syncthreads();

  // Get the starting points for the merge for A and B. Why do we subtract 1
  // from B? At the start of the array, we are pointing to output 0 and 
  // segment 0. But we don't really start merging A until we've encountered
  // its start flag at B. That is, the first iteration should increment b_index
  // to 0, then start merging from the first segment of A, so b_index needs to
  // start at -1.
  int a_index = range.a_begin + mp;
  int b_index = range.b_begin + (diag - mp) - 1;

  return lbs_placement_t {
    range, a_index, b_index
  };
}

struct lbs_fill_t {
  merge_range_t range;
  int b_offset;
};

template<int nt, int vt, typename segments_it, typename partition_it>
MGPU_DEVICE lbs_fill_t cta_load_balance_fill(int count, 
  segments_it segments, int num_segments, int tid, int cta, 
  partition_it partitions, int* shared) {
 
  merge_range_t range = compute_merge_range(count, num_segments, cta, 
    nt * vt, partitions[cta], partitions[cta + 1]);

  int* a_shared = shared - range.a_begin;
  int* b_shared = shared + range.a_count();

  lbs_placement_t placement = cta_load_balance_place<nt, vt>(tid, range, 
    count, segments, num_segments, b_shared);

  // Adjust the b pointer by the loaded b_begin. This lets us dereference it
  // directly with the segment index.
  b_shared -= placement.range.b_begin;

  // Fill shared memory with the segment IDs of the in-range values.
  int cur_item = placement.a_index;
  int cur_segment = placement.b_index;

  iterate<vt>([&](int i) {
    bool p = cur_item < b_shared[cur_segment + 1];
    if(p) a_shared[cur_item++] = cur_segment;
    else ++cur_segment;
  });
  __syncthreads();

  return lbs_fill_t {
    range,
    range.a_count() - placement.range.b_begin
  };
}

template<int nt, int vt>
struct cta_load_balance_t {
  enum { nv = nt * vt };
  struct storage_t {
    int indices[nv + 2];
  };

  struct result_t {
    lbs_placement_t placement;
    merge_range_t merge_range;

    // thread-order data.
    int merge_flags;

    // strided-order data.
    array_t<int, vt> indices;
    array_t<int, vt> segments;
    array_t<int, vt> ranks;
  };

  template<typename segments_it, typename partition_it>
  MGPU_DEVICE result_t load_balance(int count, segments_it segments, 
    int num_segments, int tid, int cta, partition_it partitions, 
    storage_t& storage) const {

    merge_range_t range = compute_merge_range(count, num_segments, cta, 
      nv, partitions[cta], partitions[cta + 1]);

    int* a_shared = storage.indices - range.a_begin;
    int* b_shared = storage.indices + range.a_count();

    lbs_placement_t placement = cta_load_balance_place<nt, vt>(tid, range, 
      count, segments, num_segments, b_shared);

    // Adjust the b pointer by the loaded b_begin. This lets us dereference it
    // directly with the segment index.
    b_shared -= placement.range.b_begin;

    // Store the segment of each element in A.
    int cur_item = placement.a_index;
    int cur_segment = placement.b_index;
    int merge_flags = 0;

    // Fill shared memory with the segment IDs of the in-range values.
    iterate<vt + 1>([&](int i) {
      // Compare the output index to the starting position of the next segment.
      bool p = cur_item < b_shared[cur_segment + 1];
      if(p && i < vt) // Advance A (the needle). 
        a_shared[cur_item++] = cur_segment;
      else  // Advance B (the haystack)
        ++cur_segment;
      merge_flags |= (int)p<< i;
    });
    __syncthreads();

    // Load the segment indices in strided order. Use the segment ID to compute
    // rank of each element. These strided-order (index, seg, rank) tuples
    // will be passed to the lbs functor.
    array_t<int, vt> indices, seg, ranks;
    iterate<vt>([&](int i) {
      int j = nt * i + tid;
      indices[i] = range.a_begin + j;
      if(j < range.a_count()) {
        seg[i] = storage.indices[j];
        ranks[i] = indices[i] - b_shared[seg[i]];
      } else {
        seg[i] = range.b_begin;
        ranks[i] = -1;
      }
    });
    __syncthreads();

    return result_t { 
      placement, range, merge_flags,
      indices, seg, ranks
    };
  }
};


namespace detail {

template<int nt, typename pointers_t>
struct cached_segment_load_t {

  enum { size = tuple_size<pointers_t>:: value };
  typedef make_index_sequence<size> seq_t;
  typedef tuple_iterator_value_t<pointers_t> value_t;

  template<typename seq_t>
  struct load_storage_t;

  template<size_t... seq_i>
  struct load_storage_t<index_sequence<seq_i...> > {
    tuple<
      array_t<typename tuple_element<seq_i, value_t>::type, nt>...
    > data;

    MGPU_HOST_DEVICE void store_value(const value_t& value, int index) {
      swallow(get<seq_i>(data)[index] = get<seq_i>(value)...);
    }

    MGPU_HOST_DEVICE value_t load_value(int index) const {
      return make_tuple(get<seq_i>(data)[index]...);
    }
  };

  typedef load_storage_t<seq_t> storage_t;

  template<int vt0, int vt>
  MGPU_DEVICE static array_t<value_t, vt> load(int tid, int count,
    range_t range, array_t<int, vt> segments, storage_t& storage, 
    pointers_t iterators) {
    
    array_t<value_t, vt> loaded;
    if(range.count() <= nt) {
      // Cached load through shared memory.
      if(tid < range.count()) {
        value_t value = mgpu::load(iterators, range.begin + tid);
        storage.store_value(value, tid);
      }
      __syncthreads();

      // Load the values into register.
      strided_iterate<nt, vt, vt0>([&](int i, int j) {
        loaded[i] = storage.load_value(segments[i] - range.begin);
      }, tid, count);
      __syncthreads();

    } else {
      // Direct load.
      strided_iterate<nt, vt, vt0>([&](int i, int j) {
        loaded[i] = mgpu::load(iterators, segments[i]);      
      }, tid, count);
    }

    return loaded;
  }
};

template<int nt>
struct cached_segment_load_t<nt, tuple<> > {
  typedef empty_t storage_t;
  typedef tuple<> value_t;

  template<int vt0, int vt>
  MGPU_DEVICE static array_t<value_t, vt> load(int tid, int count,
    range_t range, array_t<int, vt> segments, storage_t& storage,
    tuple<> iterators) {

    return array_t<value_t, vt>();
  }
};

} // namespace detail 

END_MGPU_NAMESPACE
