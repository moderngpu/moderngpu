# API Reference

moderngpu includes functions that operate on iterators and functions that operate on user-provided lambdas. Often these are just two sides of the same coin--you can adapt an iterator to a lambda-accepting function by dereferencing the iterator, and you can adapt a lambda to an iterator-accepting function by using the library's `make_load_iterator` and `make_store_iterator` functions to wrap the lambda. 

Many of the lambda-accepting functions are called _transforms_. They vary in sophistication, but each of them is useful in solving real-world problems. The transforms are very expressive, and several of moderngpu's library routines are just calls to one or more transforms specialized over simple lambdas. With this set of tools it's possible to build an entire GPU-accelerated application without writing any of your own kernels.

## Low-level transforms

#### transform 
**`transform.hxx`**
```cpp
template<
  size_t nt = 128,        // num threads per CTA. blockDim.x.
  typename func_t         // implements void operator()(int index).
>
void transform(func_t f, size_t count, context_t& context);
```
`transform` simply wraps the CUDA launch chevrons `<<< >>>` and calls a function on each of the `count` threads, to which it passes the global index of the thread, `threadIdx.x + blockIdx.x * blockDim.x`.

#### cta_launch           
**`transform.hxx`**
```cpp
template<                 
  typename launch_box,     // structure providing nt.
  typename func_t          // implements void operator()(int tid, int cta).
>
void cta_launch(func_t f, int num_ctas, context_t& context);

// Specialize over nt directly.
template<int nt, typename func_t>
void cta_launch(func_t f, int num_ctas, context_t& context);

// Dynamically loop over the number of CTAs in num_ctas.
template<typename launch_box, typename func_t>
void cta_launch(func_t f, const int* num_ctas, context_t& context);s
```
`cta_launch` is the next level of transform. Rather than launching individual threads, it creates _cooperative thread arrays_ (CTAs, also known, less descriptively, as _thread blocks_). This transformer takes as a template argument the size of the CTA in threads. This compile-time constant `nt` can be used for sizing shared memory and defining structural aspects in the code, such as the number of iterations in an unrolled loop. The caller specifies the number of CTAs to launch, and the provided lambda is invoked on each thread in each of those CTAs. 

#### cta_transform
**`transform.hxx`**
```cpp
template<
  typename launch_box,     // structure providing (nt, vt).
  typename func_t          // implements void operator()(int tid, int cta).
>
void cta_transform(func_t f, int count, context_t& context);

template<int nt, int vt = 1, typename func_t>
void cta_transform(func_t f, int count, context_t& context);
```
`cta_transform` is the workhorse transformer of moderngpu 2.0. It's used internally to implement most of the library's high-level transforms. The caller specifies a `count` of work-items and the transformer creates enough CTAs to cover them. Each CTA processes `nt * vt` work-items. `nt` is the number of threads per CTA. `vt` is the _grain size_ of the kernel, or the number of work-items processed per thread. These constants are typically loaded from a _launch box_ which specifies launch parameters on a per-architecture basis.

The provided kernel function `f` is called once per thread in each of the launched CTAs. Handling fractional tiles is the user's responsibility. 

## High-level transforms

#### transform_reduce
**`kernel_reduce.hxx`**
```cpp
template<
  typename launch_arg_t = empty_t,  // optional launch_box overrides default.
  typename func_t,         // implements type_t operator()(int index).
  typename output_it,      // iterator to output. one output element.
  typename op_t            // reduction operator implements
                           //   type_t operator()(type_t left, type_t right).
>
void transform_reduce(func_t f, int count, output_it reduction, op_t op, 
  context_t& context);
```
`transform_reduce` invokes the caller-supplied lambda and adds together all the results. This is the same parallel reduction algorithm included in every CUDA computing library, but is presented here with more convenient lambda-centric packaging.

#### transform_scan
**`kernel_scan.hxx`**
```cpp
template<
  scan_type_t scan_type = scan_type_exc, // inclusive or exclusive scan.
  typename launch_arg_t = empty_t,
  typename func_t,         // implements type_t operator()(int index).
  typename op_t,           // reduction operator implements 
                           //   type_t operator()(type_t a, type_t b).
void transform_scan(func_t f, int count, output_it output, op_t op,
  reduction_it reduction, context_t& context);
```
`transform_scan` invokes the provided lambda and computes a prefix sum over all elements. The reduction is also returned. In fact, by utilizing the _lambda iterator_ feature in moderngpu 2.0, you can get the reduction or even the output of the scan itself passed to additional `__device__`-tagged lambdas that you provide. The distinction between _iterators_ (a generalization of pointers) and _functions_ is mostly about syntax rather than functionality.

#### transform_segreduce
**`kernel_segreduce.hxx`**
```cpp
template<
  typename launch_arg_t = empty_t, // provides constants (nt, vt, vt0).
  typename func_t,         // implements type_t operator()(int index).
  typename segments_it,    // segments-descriptor array.
                           //   specificies starting offset of each segment.
  typename output_it,      // output iterator. one output per segment.
  typename op_t,           // reduction operator implements
                           //   type_t operator()(type_t a, type_t b)
  typename type_t
>
void transform_segreduce(func_t f, int count, segments_it segments, 
  int num_segments, output_it output, op_t op, type_t init, 
  context_t& context);
```
`transform_segreduce` reduces irregularly-sized sequences of values, called segments, using a reduction operator (such as `mgpu::plus_t<>`). An initialization value is provided for the reduction of empty segments. Segments are described by an integer array where each element indicates the beginning of that segment within a single flattened view of the operation. The first segment starts at 0 (so segments[0] = 0), and each subsequent segments-descriptor element is advanced by the size of the previous segment. That is, the segments descriptor is the exclusive prefix sum of the segment sizes. This is a common convention, and is known in graph and sparse matrix literature as the [compressed sparse row](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_.28CSR.2C_CRS_or_Yale_format.29) format.

Functions that use load-balancing search internally, such as `transform_segreduce`, specialize over a `vt0` parameter in the launch parameters. If not specified, the launch box mechanism defaults `vt0` to `vt`. `vt0` is a caching term which specifies the number of unconditional loads to execute per thread. Segment geometries with more values per segment will benefit from increasing `vt0`, while geometries with smaller segments may benefit from decreasing `vt0`. A rule of thumb is to set vt0 to 
```
vt0 = floor(vt * count / (count + num_segments))
```
if this ratio can be reasonably estimated in advance.

#### transform_lbs
**`kernel_load_balance.hxx`**
```cpp
template<
  typename launch_arg_t = empty_t, // provides (nt, vt, vt0)
  typename func_t,         // load-balancing search callback implements
                           //   void operator()(int index,   // work-item
                           //                   int seg,     // segment ID
                           //                   int rank,    // rank within seg
                           //                   tuple<...> cached_values).
  typename segments_it,    // segments-descriptor array.
  typename tpl_t           // tuple<> of iterators for caching loads.
>
void transform_lbs(func_t f, int count, segments_it segments, 
  int num_segments, tpl_t caching_iterators, context_t& context);

// version of transform_lbs without caching iterators
template<
  typename launch_arg_t = empty_t, 
  typename func_t,         // load-balancing search callback implements
                           //   void operator()(int index, int seg, int rank).
  typename segments_it
>
void transform_lbs(func_t f, int count, segments_it segments, 
  int num_segments, context_t& context);  
```
`transform_lbs` is the new front-end to the _load-balancing search_ pattern introduced in moderngpu 1.0. The caller describes a collection of irregularly-sized _segments_ with an array that indexes into the start of each segment. (This is identical to the prefix sum of the segment sizes.) Load-balancing search restores shape to this flattened array by calling your lambda once for each element in each segment and passing the segment identifier and the rank of the element within that segment. Unlike ad hoc scheduling methods which assign a thread, or a warp, or a thread-block to a segment, _load-balancing search_ is robust with respect to the shape of the problem; you can bet on consistently good performance for all conditions. This is a truly powerful mechanism and a surprising number of useful functions can be defined with one-line lambdas using `transform_lbs`. Consider it the data science _mapper_ of CUDA programs.

#### lbs_segreduce
**`kernel_segreduce.hxx`**
```cpp
template<
  typename launch_arg_t = empty_t, // provides (nt, vt, vt0)
  typename func_t,         // load-balancing search callback implements
                           //   type_t operator()(int index,   // work-item
                           //                     int seg,     // segment ID
                           //                     int rank,    // rank within seg
                           //                     tuple<...> cached_values).
  typename segments_it,    // segments-descriptor array.
  typename tpl_t,          // tuple<> of iterators for caching loads.
  typename output_it,      // output iterator. one output per segment.
  typename op_t,           // reduction operator implements
                           //   type_t operator()(type_t a, type_t b).
  typename type_t
>
void lbs_segreduce(func_t f, int count, segments_it segments,
  int num_segments, tpl_t caching_iterators, output_it output, op_t op,
  type_t init, context_t& context);

// version of lbs_segreduce without caching iterators.
template<
  typename launch_arg_t = empty_t, 
  typename func_t,         // load-balancing search callback implements
                           //   type_t operator()(int index, int seg, int rank).
  typename segments_it, 
  typename output_it, 
  typename op_t,
  typename type_t
>
void lbs_segreduce(func_t f, int count, segments_it segments,
  int num_segments, output_it output, op_t op, type_t init, 
  context_t& context);
```
`lbs_segreduce` combines `transform_lbs` with segmented reduction. The caller's lambda is invoked with the segment ID and rank for each element, but the return values are folded into a single element per segment. `lbs_segreduce` is like a simultaneous mapper and reducer: each segment is expanded into an irregular number of work-items; a user-provided function is called on each work-item; then the results are reduced according to the segment geometry. This is the most sophisticated mechanism in moderngpu 2.0, and makes data science operations very easy to program.

#### transform_compact
**`kernel_compact.hxx`**
```cpp
template<typename launch_arg_t = empty_t>
stream_compact_t<launch_arg_t> 
transform_compact(int count, context_t& context);

template<typename launch_arg_t>
struct stream_compact_t {
  ...
public:
  // upsweep of stream compaction. 
  // func_t implements bool operator(int index);
  // The return value is flag for indicating that we want to *keep* the data
  // in the compacted stream.
  template<typename func_t>
  int upsweep(func_t f);

  // downsweep of stream compaction.
  // func_t implements void operator(int dest_index, int source_index).
  // The user can stream from data[source_index] to compacted[dest_index].
  template<typename func_t>
  void downsweep(func_t f);
};
```
`transform_compact` is a two-pass pattern for space-efficient stream compaction. The user constructs a `stream_compact_t` object by calling `transform_compact`. On the upsweep, the user provides a lambda function which returns `true` to retain the specified `index` in the streamed output. On the downsweep, the user implements a `void`-returning lambda which takes the index to stream to and the index to stream from. The user may implement any element-wise copy or transformation in the body of this lambda.

**`test_compact.cu`**
```cpp
    // Construct the compaction state with transform_compact.
    auto compact = transform_compact(count, context);

    // The upsweep determines which items to compact.
    int stream_count = compact.upsweep([]MGPU_DEVICE(int index) {
      // Return true if we want to keep this element.
      // Only stream if the number of bits in index is a multiple of 5.
      bool keep = 0 == (popc(index) % 5);
      return keep;
    });

    // Compact the results into this buffer.
    mem_t<int> special_numbers(stream_count, context);
    int* special_numbers_data = special_numbers.data();
    compact.downsweep([=]MGPU_DEVICE(int dest_index, int source_index) {
      special_numbers_data[dest_index] = source_index;
    });
```
`test_compact.cu` demonstrates a simple usage of moderngpu's stream compaction. In this sample, all indices with a multiple of 5 bits set in the index's binary representation are compacted. The first statement uses `transform_compact` to construct a `stream_compact_t` object; this contains temporary state to communicate compaction information between CTAs. 

The `upsweep` call is a lambda that returns `true` to keep the index and `false` to discard it. In this example, the lambda does not capture any variables from the enclosing scope, but it may. The return value is the total number of values to compact. The client uses this to allocate exactly enough memory to hold the streamed outputs.

In the `downsweep` call the lambda is provided both the index to stream to and the index to stream from. In this example, the lambda simply stores the source index into the compacted array, but any operation is admitted.

[stream compaction in thrust:](https://thrust.github.io/doc/group__stream__compaction.html)
```cpp
template<typename InputIterator1 , typename InputIterator2 , typename OutputIterator , typename Predicate >
OutputIterator thrust::copy_if (InputIterator1 first, InputIterator1 last, InputIterator2 stencil, OutputIterator result, Predicate pred);

template<typename InputIterator , typename OutputIterator , typename T >
OutputIterator  thrust::remove_copy (InputIterator first, InputIterator last, OutputIterator result, const T &value)

template<typename InputIterator , typename OutputIterator , typename Predicate >
OutputIterator  thrust::remove_copy_if (InputIterator first, InputIterator last, OutputIterator result, Predicate pred);

template<typename InputIterator , typename OutputIterator >
OutputIterator  thrust::unique_copy (InputIterator first, InputIterator last, OutputIterator result);
...
```
This explicit two-pass process is better than the stream compaction functionality available in [thrust](https://thrust.github.io/doc/group__stream__compaction.html). In those functions the size of the compacted output is returned to the caller only after the stream compaction has completed. (It is implied by the position of the return iterator minus `result`). The user may need to _over-allocate_ storage for the results, conservatively allocating enough space to copy the entire input array into the output.

thrust also defines dozens of related stream compactions such as `copy_if`, `remove_copy_if`, `unique_copy`, `unique_by_key_copy`, etc. These cases are all handled by `transform_compact`'s `upsweep` lambda. `copy_if` has behavior like our test case, testing a single value and returning `true` to keep it. 

`unique_copy` keeps only the first element in a group of duplicate elements. A lambda can implement this behavior by simply returning `!index || data[index] != data[index - 1]`. That is, an element is streamed if and only if it has a different value from the preceding element.

moderngpu provides more coverage of functionality with many fewer functions than other libraries by recognizing different categories of operations and allowing the user to specialize behavior.

#### lbs_workcreate
**`kernel_workcreate.hxx`**
```cpp
namespace expt {

// Use lbs_workcreate to construct an expt::workcreate_t instance. Then call
// upsweep and downsweep, providing an appropriate lambda function.
template<typename launch_arg_t = empty_t, typename segments_it>
workcreate_t<launch_arg_t, segments_it>
lbs_workcreate(int count, segments_it segments, int num_segments,
  context_t& context) {
  return workcreate_t<launch_arg_t, segments_it> {
    count, segments, num_segments, context
  };
}

template<typename launch_arg_t, typename segments_it>
struct workcreate_t {
  ...
public:
  struct count_t {
    int count;
    int num_segments;
  };

  // f(int index, int seg, int rank, tuple<...> desc) returns the number
  // of work-items to create.
  template<typename func_t, typename tpl_t>
  count_t upsweep(func_t f, tpl_t caching_iterators);

  // upsweep without caching iterators.
  template<typename func_t>
  count_t upsweep(func_t f);

  // f(int dest_seg, int index, int source_seg, int rank, tuple<...> desc)
  // returns the number of work-items to create.
  template<typename func_t, typename tpl_t>
  mem_t<int> downsweep(func_t f, tpl_t caching_iterators);

  template<typename func_t>
  mem_t<int> downsweep(func_t f);
};

} // namespace expt
```
`lbs_workcreate` returns the state for a two-pass load-balancing search work-creation process. This function is included in the `mgpu::expt` namespace to reflect its experimental status.

`lbs_workcreate` is called similarly to `transform_lbs`. The work-item count and segments-descriptor array are passed as arguments, and have the same meaning as in `transform_lbs`. However these individual work-items now have the capability of _dynamic work creation_. This function returns a `workcreate_t` object, on which `upstream` and `downstream` calls are made by the client. `upsweep` returns a `count_t` structure with the number of total work-items and segments generated.

1. `upsweep` is called for every work-item in the work load and returns the number of work-items to create. If the return count is greater than 0, one new _segment_ of work will is emitted. That segment will have _count_ number of ranks associated with it. This call can be non-deterministic. The [improved breadth-first search](#improved-breadth-first-search) demo uses the CUDA intrinsic `atomicCAS` to non-deterministically generate work for newly-visited vertices.
2. `downsweep` once again returns the number of work-items to create (it must match what it returned in `upsweep`), however `downsweep` is only invoked on work-items that attempted to generate new work in `upstream`. During this second pass the callback lambda `f` is also provided with the index of the newly-created segment of work. The caller should stream to this index information for processing the work in this segment. The function returns the segments-descriptor array of the dynamically-created work.

Because the implementation has no knowledge of the number of work-items opting for dynamic work creation it cannot size the segments-descriptor array; nor does it choose to store the work-creation requests for each work-item. It therefore makes two passes. The first pass requests the number of work-items created. This is scanned on a per-CTA basis, and passed to the second pass, which again requests the number of work-items created, adds that to the running total of work-items, and stores to the new segments-descriptor array.

This pattern is constructed by fusing together the operations of load-balancing search, stream compaction, and scan to enable the user to dispatch work-items and dynamically create new ones.

## Array functions

#### reduce
**`kernel_reduce.hxx`**
```cpp
template<typename launch_arg_t = empty_t, typename input_it, 
  typename output_it, typename op_t>
void reduce(input_it input, int count, output_it reduction, op_t op, 
  context_t& context);
```

#### scan
**`kernel_scan.hxx`**
```cpp

template<scan_type_t scan_type = scan_type_exc,
  typename launch_arg_t = empty_t, typename input_it, 
  typename output_it, typename op_t, typename reduction_it>
void scan_event(input_it input, int count, output_it output, op_t op, 
  reduction_it reduction, context_t& context, cudaEvent_t event);

template<scan_type_t scan_type = scan_type_exc, 
  typename launch_arg_t = empty_t, typename input_it, 
  typename output_it, typename op_t, typename reduction_it>
void scan(input_it input, int count, output_it output, op_t op, 
  reduction_it reduction, context_t& context);

template<scan_type_t scan_type = scan_type_exc, 
  typename launch_arg_t = empty_t, 
  typename input_it, typename output_it>
void scan(input_it input, int count, output_it output, context_t& context);

template<scan_type_t scan_type = scan_type_exc, 
  typename launch_arg_t = empty_t, typename func_t, typename output_it,
  typename op_t, typename reduction_it>
void transform_scan_event(func_t f, int count, output_it output, op_t op,
  reduction_it reduction, context_t& context, cudaEvent_t event);

template<
  scan_type_t scan_type = scan_type_exc, // scan_type_exc or scan_type_inc.
  typename launch_arg_t = empty_t, 
  typename func_t,         // implements type_t operator()(int index).
  typename output_it,
  typename op_t,           // implements type_t operator()(type_t a, type_t b).   
  typename reduction_it    // iterator for storing the scalar reduction
                           //   pass discard_iterator_t<type_t>() to ignore 
                           //   the reduction.
>
void transform_scan(func_t f, int count, output_it output, op_t op,
  reduction_it reduction, context_t& context);
```
The prefix sum implementation `scan` has several overloads for convenience. The scan is implemented as three kernels: an upsweep that reduces values mapped into tiles; a spine which scans the partial reductions and stores the total reduction of the array; and a downsweep which distributes the scanned partials and completes the scan. 

Because of this structure, the reduction is computed about a third of the way through the reduction. (One load for the upsweep; one load and one store for the downsweep.) The caller may need the reduction as soon as possible to allocate memory and schedule more work. Rather than suffer the latency of waiting for `scan` to return and copying the reduction from device to host memory, the user can allocate space in page-locked host memory and call `scan_event` or `transform_scan_event` to store the reduction directly into this host memory. The caller provides an event which is recorded when the reduction is computed. When `scan_event` returns the caller can synchronize on the event and read the reduction directly out of page-locked memory. This relieves the code from waiting for the downsweep kernel to complete before getting the reduction.

#### merge

**`kernel_merge.hxx`**
```cpp
// Key-value merge.
template<
  typename launch_arg_t = empty_t,
  typename a_keys_it, typename a_vals_it, // A source keys and values
  typename b_keys_it, typename b_vals_it, // B source keys and values
  typename c_keys_it, typename c_vals_it, // C dest keys and values
  typename comp_t         // implements bool operator()(type_t a, type_t b)
                          //   computes a < b.
>
void merge(a_keys_it a_keys, a_vals_it a_vals, int a_count, 
  b_keys_it b_keys, b_vals_it b_vals, int b_count,
  c_keys_it c_keys, c_vals_it c_vals, comp_t comp, context_t& context);

// Key-only merge.
template<typename launch_t = empty_t,
  typename a_keys_it, typename b_keys_it, typename c_keys_it,
  typename comp_t>
void merge(a_keys_it a_keys, int a_count, b_keys_it b_keys, int b_count,
  c_keys_it c_keys, comp_t comp, context_t& context);
```

#### sorted_search

**`kernel_sortedsearch.hxx`**
```cpp
template<
  bounds_t bounds,         // bounds_lower or bounds_upper
  typename launch_arg_t = empty_t,
  typename needles_it,     // search to find the insertion index of each
  typename haystack_it,    //   sorted needle into the sorted haystack.
  typename indices_it,     // output integer indices. 
                           //   sized to the number of needles.
  typename comp_it         // implements bool operator()(type_t a, type_t b)
                           //   computes a < b.
>
void sorted_search(needles_it needles, int num_needles, haystack_it haystack,
  int num_haystack, indices_it indices, comp_it comp, context_t& context);
```
`sorted_search` is a vectorized searching algorithm. It's equivalent to calling `std::lower_bound` or `std::upper_bound` into the haystack array for each key in the needles array. We could easily implement a parallel binary search on the GPU, and it would have cost O(A log B).

But we can do better if the keys in the needles array are themselves sorted. Rather than process each needle individually using a binary search, the threads cooperatively compute the entire output array `indices` as a single merge-like operation. Consider comparing the front of the haystack array to the front of the needles array. If `*haystack < *needles`, we advance the `haystack` pointer. Otherwise we store the index of `haystack` to the `indices` results array at the `needles` index and advance the `needles` index.

The cost of `sorted_search` is O(A + B) and the routine is easily load-balanced over all available processing units. This function is helpful when implementing key-matching operations like [relational join](#relational-join).

#### bulk_insert

**`kernel_bulkinsert.hxx`**
```cpp
// Insert the values at a_keys before the values at b_keys identified by
// insert.
template<typename launch_t = empty_t, typename a_it, typename insert_it, 
  typename b_it, typename c_it>
void bulk_insert(a_it a, insert_it a_insert, int insert_size, b_it b, 
  int source_size, c_it c, context_t& context);
```

#### bulk_remove

**`kernel_bulkremove.hxx`**
```cpp
template<typename launch_arg_t = empty_t,
  typename input_it, typename indices_it, typename output_it>
void bulk_remove(input_it input, int count, indices_it indices, 
  int num_indices, output_it output, context_t& context);
```

#### mergesort

**`kernel_mergesort.hxx`**
```cpp
// Key-value sort.
template<
  typename launch_arg_t = empty_t, 
  typename key_t,          // key type.
  typename val_t,          // value type.
  typename comp_t          // implements bool operator()(type_t a, type_t b)
                           //   computes a < b.
>
void mergesort(key_t* keys_input, val_t* vals_input, int count,
  comp_t comp, context_t& context);

// Key-only sort.
template<typename launch_arg_t = empty_t, typename key_t, typename comp_t>
void mergesort(key_t* keys_input, int count, comp_t comp, 
  context_t& context);
```
Sort keys or key-value pairs. Unlike most other functions in moderngpu, this takes no iterators and transforms arrays in place.

#### segmented_sort

**`kernel_segsort.hxx`**
```cpp
// Key-value segmented sort.
template<typename launch_arg_t = empty_t, typename key_t, typename val_t,
  typename seg_it, typename comp_t>
void segmented_sort(key_t* keys_input, val_t* vals_input, int count,
  seg_it segments, int num_segments, comp_t comp, context_t& context);

// Key-value segmented sort. Automatically generate indices to sort as values.
template<typename launch_arg_t = empty_t, typename key_t, typename seg_it, 
  typename comp_t>
void segmented_sort_indices(key_t* keys, int* indices, int count, 
  seg_it segments, int num_segments, comp_t comp, context_t& context);

// Key-only segmented sort.
template<typename launch_arg_t = empty_t, typename key_t, typename seg_it, 
  typename comp_t>
void segmented_sort(key_t* keys_input, int count, seg_it segments, 
  int num_segments, comp_t comp, context_t& context);
```

Sort keys or key-value pairs in place. This function sorts elements within segments and denoted by the segments-descriptor array. `segmented_sort_indices` ignores the initial contents of `indices` and on return stores there the gather indices of the keys.

#### inner_join

**`kernel_join.hxx`**
```cpp
template<typename launch_arg_t = empty_t, 
  typename a_it, typename b_it, typename comp_t>
mem_t<int2> inner_join(a_it a, int a_count, b_it b, int b_count, 
  comp_t comp, context_t& context);
```

#### load\_balance\_search

#### interval_move

**`kernel_intervalmove.hxx`**
```cpp
template<typename launch_arg_t = empty_t, typename input_it, 
  typename segments_it, typename output_it>
void interval_expand(input_it input, int count, segments_it segments,
  int num_segments, output_it output, context_t& context);

template<typename launch_arg_t = empty_t, typename input_it, 
  typename segments_it, typename gather_it, typename output_it>
void interval_gather(input_it input, int count, segments_it segments,
  int num_segments, gather_it gather, output_it output, context_t& context);

template<typename launch_arg_t = empty_t, typename input_it, 
  typename segments_it, typename scatter_it, typename output_it>
void interval_scatter(input_it input, int count, segments_it segments,
  int num_segments, scatter_it scatter, output_it output, context_t& context);

template<typename launch_arg_t = empty_t, 
  typename input_it, typename segments_it, typename scatter_it,
  typename gather_it, typename output_it>
void interval_move(input_it input, int count, segments_it segments,
  int num_segments, scatter_it scatter, gather_it gather, output_it output, 
  context_t& context);
```

#### inner_join

**`kernel_join.hxx`**
```cpp
template<typename launch_arg_t = empty_t, 
  typename a_it, typename b_it, typename comp_t>
mem_t<int2> inner_join(a_it a, int a_count, b_it b, int b_count, 
  comp_t comp, context_t& context);
```

#### segreduce

**`kernel_segreduce.hxx`**
```cpp
template<typename launch_arg_t = empty_t, typename input_it,
  typename segments_it, typename output_it, typename op_t, typename type_t>
void segreduce(input_it input, int count, segments_it segments, 
  int num_segments, output_it output, op_t op, type_t init, 
  context_t& context);

template<typename launch_arg_t = empty_t, typename matrix_it,
  typename columns_it, typename vector_it, typename segments_it, 
  typename output_it>
void spmv(matrix_it matrix, columns_it columns, vector_it vector,
  int count, segments_it segments, int num_segments, output_it output,
  context_t& context);
```
