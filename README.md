
# moderngpu 2.0 

**(c) 2016 [Sean Baxter](http://twitter.com/moderngpu)** 

**You can drop me a line [here](mailto:moderngpu@gmail.com)**

moderngpu is a productivity library for general-purpose computing on GPUs. It is a header-only C++ library written for CUDA. The unique value of the library is in its accelerated primitives for solving irregularly parallel problems. 

The notes below document the interface and describe the usage of the library's host-callable functions. In these operations the kernel launch is made moderngpu, although the caller has much flexibility in customizing the behavior of the function and tuning its performance. 

The library also includes many CTA-level structures which the higher-order host-callable functions are built upon. Appealing to a smaller audience of developers and undergoing a higher rate of churn, these functions are mostly undocumented at this time, although experienced programmers should be able to understand the usage of these features upon inspection of the source. However, unlike with moderngpu 1.0, effective use of CTA-level structures is not critical to realizing most of the value of the library.

The distribution includes two demos on the usage of the high-level transforms:
  1. **[bfs.cu](#breadth-first-search)** - a simple implementation of a classic breadth-first search graph algorithm.
  2. **[cities.cu](#attu-station-ak)** - a geographic data sciences query posed as high-level segmented transformations.

Five tutorials on kernel-level programming are also included:
  1. **[tut_01_transform.cu](#tutorial-1---parallel-transforms)** - launch a parallel for loop using lambda closure to capture kernel arguments.
  2. **[tut_02_cta_launch.cu](#tutorial-2---cooperative-thread-arrays)** - launch a cooperatively-parallel kernel using `cta_launch`. 
  3. **[tut_03_launch_params.cu](#tutorial-3---launch-box)** - specify architecture-specific tuning parameters using launch box.
  4. **[tut_04_launch_custom.cu](#tutorial-4---custom-launch-parameters)** - architecture-specific tuning with a custom parameters structure.
  5. **[tut_05_iterators.cu](#tutorial-5---iterators)** - attach lambda functions to load and store operators.

Users familiar with CUDA programming wishing to cut to the chase should start at [Examples of automatic load-balancing](#examples-of-automatic-load-balancing) where the most novel features of this library are demonstrated. 

## Contents
1. [Release notes](#release-notes)
  1. [License](#license)
1. [Introduction](#introduction)
1. [The power of transforms](#the-power-of-transforms)
  1. [Low-level transforms](#low-level-transforms)
    1. [transform](#transform)
    1. [cta_launch](#cta_launch)
    1. [cta_transform](#cta_transform)
  1. [High-level transforms](#high-level-transforms)
    1. [transform_reduce](#transform_reduce)
    1. [transform_scan](#transform_scan)
    1. [transform_segreduce](#transform_segreduce)
    1. [transform_lbs](#transform_lbs)
    1. [lbs_segreduce](#lbs_segreduce)
  1. [Array functions](#array-functions)
    1. [reduce](#reduce)
    1. [scan](#scan)
    1. [merge](#merge)
    1. [sorted_search](#sorted_search)
    1. [bulk_insert](#bulk_insert)
    1. [bulk_remove](#bulk_remove)
    1. [mergesort](#mergesort)
    1. [segsort](#segsort)
    1. [inner_join](#inner_join)
    1. [load_balance_search](#load_balance_search)
    1. [interval_move](#interval_move)
    1. [inner_join](#inner_join)
    1. [segreduce](#segreduce)
1. [Examples of automatic load-balancing](#examples-of-automatic-load-balancing)
  1. [Sparse matrix*vector](#sparse-matrix--vector)
  1. [Interval move and interval expand](#interval-move-and-interval-expand)
  1. [Relational join](#relational-join)
  1. [Breadth-first search](#breadth-first-search)
  1. [Attu Station, AK](#attu-station-ak)
1. [Kernel programming](#kernel-programming)
  1. [Tutorial 1 - parallel transforms](#tutorial-1---parallel-transforms)
  1. [Tutorial 2 - cooperative thread arrays](#tutorial-2---cooperative-thread-arrays)
  1. [Tutorial 3 - launch box](#tutorial-3---launch-box)
  1. [Tutorial 4 - custom launch parameters](#tutorial-4---custom-launch-parameters)
  1. [Tutorial 5 - iterators](#tutorial-5---iterators)
1. [Usage tips](#usage-tips)

## Release notes
```
2.00 2016 Mar 28 - 
  moderngpu 2.0 first release.

  Everything rewritten.
  
  Use -std=c++11 and --expt-extended-lambda to build.
  
  Developed and tested with CUDA 7.5.17, g++ 4.8.4 on 64-bit linux.

  Tests and samples build on OSX clang-700-1.81.
  
  Invocations of these kernels with certain arguments can push Visual 
  Studio 2013 to the breaking point. "Exceeded token length" is a common
  error when passing iterators to high-level transforms. Users may be better
  off avoiding Visual Studio until CUDA 8.0 is released with VS 2015 support.

  Visual Studio 2013 has broken constexpr support, so with that version
  moderngpu chooses to redefine its constexpr functions as macros, injecting
  them into the global scope. Sorry!

  TODO: Complete kernel unit tests. 
  TODO: API references.
  TODO: Multiset operations for feature parity with moderngpu 1.0.
  TODO: Restore the moderngpu 1.0 text by integrating it with this markdown.
```

### License
> Copyright (c) 2016, Sean Baxter
> All rights reserved.
> 
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
> 
> 1. Redistributions of source code must retain the above copyright notice, this
>    list of conditions and the following disclaimer.
> 2. Redistributions in binary form must reproduce the above copyright notice,
>    this list of conditions and the following disclaimer in the documentation
>    and/or other materials provided with the distribution.
> 
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
> ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
> WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
> ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
> (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
> LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
> ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
> (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
> SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
> 
> The views and conclusions contained in the software and documentation are those
> of the authors and should not be interpreted as representing official policies,
> either expressed or implied, of the FreeBSD Project.

## Introduction
moderngpu 2.0 is a complete rewrite of the utilities and algorithms in moderngpu. It's also the first update to that library since shortly after its release in 2013.

This new version takes the algorithms from moderngpu 1.0 and combines them with new composition models to push usability to the front. The kernel implementations are compact and the interfaces are pliant. The intent is that the code is clear at every line.
 
moderngpu 2.0 is parsimonious in its implementation. The initial commit checks in at 5,000 lines of code. Compare to other popular CUDA libraries: [cub](http://nvlabs.github.io/cub) is 45,000 lines; [CUDPP](http://cudpp.github.io) is 26,000 lines; and [thrust](http://thrust.github.io) is north of 160,000 lines. In fact, moderngpu 0.5 might be a more fitting version number, as it's half the size of moderngpu 1.0. 

CUDA 7.5's solid C++11 support made a big impact on the design of this new version. Major language bullet points like lambda functions have naturally found their way into the implementation, but even minor improvements like default arguments for function templates have helped make for a tremendously cleaner design.

I hope this library will be an asset for both novice and old-school GPU programmers. The practical value of the code is in the many data-parallel _transform functions_. The transforms can all accept user-defined lambda functions (lambdas can be attached to almost anything in the library) to specialize behavior. Many of the more advanced kernels of moderngpu 1.0, including load-balancing search; interval expand, move, gather and scatter; and sparse matrix-vector multiply are now implemented as single function calls. Relational inner join, which was a complicated multi-kernel design in moderngpu 1.0, is now a short sequence of calls to canned routines capped by a use of the load-balancing search.

This library also carries a certain philosophical message: Software is an asset, code a liability. On good days I'd add 200 or 300 lines to this repository. On great days I'd subtract 500. Keeping your code small means you'll be more ready to discard it when better ideas come along. Computers are meant to serve ideas, but the exigencies of programming, especially in the arcane field of HPC, make code a sunk cost that can prevent an individual or an organization from sloughing off what it doesn't need and moving nimbly to the next great new thing. I built moderngpu 2.0 as the ideal toolkit for my own needs: it's expressive, it's light weight, and if you can break a problem into simple pieces, it'll help you realize your own solution.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_Sean Baxter_
 
## The power of transforms

moderngpu includes functions that operate on iterators and functions that operate on user-provided lambdas. Often these are just two sides of the same coin--you can adapt an iterator to a lambda-accepting function by dereferencing the iterator, and you can adapt a lambda to an iterator-accepting function by using the library's `make_load_iterator` and `make_store_iterator` functions to wrap the lambda. 

Many of the lambda-accepting functions are called _transforms_. They vary in sophistication, but each of them is useful in solving real-world problems. The transforms are very expressive, and several of moderngpu's library routines are just calls to one or more transforms specialized over simple lambdas. With this set of tools it's possible to build an entire GPU-accelerated application without writing any of your own kernels.

### Low-level transforms

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

### High-level transforms

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

### Array functions

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

These functions are ready-to-use but not yet documented.

#### bulk_remove

#### mergesort

#### segsort

#### inner_join

#### load\_balance\_search

#### interval_move

#### inner_join

#### segreduce

## Examples of automatic load-balancing

In the past CUDA programmers were responsible for decomposing their own problem over the regular grid of GPU threads. Let's say you need to reduce values from many different non-uniformly sized segments. Which processing units load and reduce which values? Do you assign one thread to a segment, or a warp's worth or a block's worth of threads? If you settle on a vector width and the variance of the segment size is large, do you waste vector processors by assigning small segments to a large width; do you create load-imbalance by oversubscribing large segments to too-small vectors?

moderngpu 1.0 introduced the _load-balancing search_ algorithm for dynamically mapping work to processors in a way that doesn't starve any processors of work or oversubscribe work to too-few cores. However there was a considerable cost to using this mechanism: you had to write your own kernel, allocate shared memory for the load-balancing search, and communicate between threads. In the current version, the composition model has changed, and the high-level transformers take care of all of this. You just provide a lambda and the library calls it with the desired segment information.

### Sparse matrix * vector 
Features demonstrated:

1. `transform_segreduce`

**`kernel_segreduce.hxx`**
```cpp
template<typename launch_arg_t = empty_t, typename matrix_it,
  typename columns_it, typename vector_it, typename segments_it, 
  typename output_it>
void spmv(matrix_it matrix, columns_it columns, vector_it vector,
  int count, segments_it segments, int num_segments, output_it output,
  context_t& context) { 

  typedef typename std::iterator_traits<matrix_it>::value_type type_t;

  transform_segreduce<launch_arg_t>([=]MGPU_DEVICE(int index) {
    return matrix[index] * ldg(vector + columns[index]);    // sparse m * v.
  }, count, segments, num_segments, output, plus_t<type_t>(), 
   (type_t)0, context);
}
```
In moderngpu 1.0 the sparse matrix * dense vector operator, `spmv`, needed a lot of custom programming. But in moderngpu 2.0 it's just an expression of the `transform_segreduce` pattern. The user provides a CSR-style segments-descriptor array and an MGPU_DEVICE-tagged lambda to the transformer. This lambda is called once per input element, where it loads one non-zero matrix element, looks up the column index of that matrix element, gathers a vector element at that location, and returns the product. 

`transform_segreduce` is a very intelligent transformer. It automatically distributes the work of function evaluation over the regular grid of GPU threads in a geometry-oblivious way. Matrices with a very large number of non-zero elements per row parallelize just as well as much sparser matrices. Matrices with a large variance in the number of non-zero elements per row parallelize just as well as more uniform systems.

This new `spmv` function is efficient, robust, flexible and tunable. Matrices with empty rows are handled just fine--the empty rows of the output are plugged with 0 (although you could pass in any value). The `__ldg` CUDA intrinsic is used on `sm_35` and later architectures, but only when `vector` is a pointer to an arithmetic type, so you can still pass in an iterator and generate the vector values on-the-fly. 

The tuning of the function is also pushed out to the user: pass a `launch_arg_t` to specialize the kernel over an architecture-specific set of launch parameters, including the block size, grain size, cache ratio and SM occupancy. By moving tuning parameters _outside_ of the function implementation, the user can tweak the parameters for best performance for their specific needs, including generating multiple versions of the same function for inputs with different characteristics.

### Interval move and interval expand

Features demonstrated:

1. `transform_lbs`

Interval expand and interval move were introduced in moderngpu 1.0 and written as kernels that used the CTA-level load-balancing search components. `interval_expand` is like a batched `std::fill`; it sets all the elements in each segment with a single segment-specific value. `interval_move` is like a batched `std::copy`; for each segment it copies a sequence of data from a gather point to a scatter point.

```cpp
template<typename launch_arg_t = empty_t, typename input_it, 
  typename segments_it, typename output_it>
void interval_expand(input_it input, int count, segments_it segments,
  int num_segments, output_it output, context_t& context) {

  typedef typename std::iterator_traits<input_it>::value_type type_t;
  transform_lbs<launch_arg_t>(
    [=]MGPU_DEVICE(int index, int seg, int rank) {
      output[index] = input[seg];
    }, 
    count, segments, num_segments, context
  );
}

template<typename launch_arg_t = empty_t, 
  typename input_it, typename segments_it, typename scatter_it,
  typename gather_it, typename output_it>
void interval_move(input_it input, int count, segments_it segments,
  int num_segments, scatter_it scatter, gather_it gather, output_it output, 
  context_t& context) {

  transform_lbs<launch_arg_t>(
    [=]MGPU_DEVICE(int index, int seg, int rank) {
      output[scatter[seg] + rank] = input[gather[seg] + rank];
    }, 
    count, segments, num_segments, context
  );
}
```
Both functions are succinctly implemented with the new `transform_lbs` high-level transformer. The user-provided lambda is called once for each work-item and passed the needed context: the index of the work-item within the global operation, the segment of the work-item, and the rank of the work-item within the segment. In the case of `interval_expand`, `index = 110`, `seg = 4`, `rank = 20` means we need to fill `output[110]` with the value `input[seg]`. For `interval_move` it means we need to copy the 20th item of segment 4 from `input[gather[seg] + rank]` to `output[scatter[seg] + rank]`.

`transform_lbs` with user-provided lambdas is the natural language for segmented operations like `interval_move`. Indeed, the code is a more readable than a textual description. But there is one drawback to the implementations above; they aren't as fast as the hand-coded versions in moderngpu 1.0. You see that the fill-value `input` array is dereferenced once for each item in the same segment in `interval_expand`, and both `scatter` and `gather` are dereferenced for each item in the same segment in `interval_move`. These are redundant loads and they eat up L2 bandwidth. The hand-coded version cooperatively loaded these terms by segment ID into shared memory, then read the cached values out when storing to output or computing indexing.

Sacrificing efficiency for convenience is a hard sell, especially in HPC software. How can we cache values that are indexed per-segment and retain the convenience of the `transform_lbs` interface?

**`kernel_intervalmove.hxx`**
```cpp
template<typename launch_arg_t = empty_t, typename input_it, 
  typename segments_it, typename output_it>
void interval_expand(input_it input, int count, segments_it segments,
  int num_segments, output_it output, context_t& context) {

  typedef typename std::iterator_traits<input_it>::value_type type_t;
  transform_lbs<launch_arg_t>(
    [=]MGPU_DEVICE(int index, int seg, int rank, tuple<type_t> desc) {
      output[index] = get<0>(desc);
    }, 
    count, segments, num_segments, make_tuple(input), context
  );
}

template<typename launch_arg_t = empty_t, 
  typename input_it, typename segments_it, typename scatter_it,
  typename gather_it, typename output_it>
void interval_move(input_it input, int count, segments_it segments,
  int num_segments, scatter_it scatter, gather_it gather, output_it output, 
  context_t& context) {

  transform_lbs<launch_arg_t>(
    [=]MGPU_DEVICE(int index, int seg, int rank, tuple<int, int> desc) {
      output[get<0>(desc) + rank] = input[get<1>(desc) + rank];
    }, 
    count, segments, num_segments, make_tuple(scatter, gather), context
  );
}
```
`transform_lbs` and the related `lbs_segreduce` functions are overloaded to take optional `mgpu::tuple<>` parameters. `make_tuple` fuses together _cacheable iterators_ pointing to per-segment data. The caching overload of `transform_lbs` loads from the cacheable iterators through shared memory and passes the values to the lambda, as its tuple-valued fourth parameter.

In the improved `interval_move` implementation, the scatter and gather offsets, which are likely `const int*` types, are fused into a `tuple<const int*, const int*>` value and passed to `transform_lbs`. This high-level transformer cooperatively loads one value per tied iterator for each segment spanned by the CTA through shared memory and out into register. These cached values are constructed into a `tuple<int, int>` and passed to the lambda. `interval_move`'s lambda uses `mgpu::get<>` to extract the scatter and gather offsets, which it adds to the work-item's rank as in the uncached implementation.

Explicit caching of per-segment values is cooperatively parallel operation, but `interval_expand` and `interval_move` include no cooperatively parallel code. They benefit from the feature by way of this improved composition model.

### Relational join
Features demonstrated:

1. `sorted_search`
1. `transform_scan`
1. `transform_lbs`

Consider table A joined with table B. We want to record the outer product of all matching keys from A and B.

| index| table A | table B     |
|------|---------|-------------|
| 0    | ape     | chicken     |
| 1    | ape     | cow         |
| 2    | kitten  | goat        |
| 3    | kitten  | kitten      |
| 4    | kitten  | kitten      |
| 5    | zebra   | tiger       |
| 6    |         | zebra       |

inner join table:

| index | A index  | A key      | B index | B key      |
|-------|----------|------------|---------|------------|
| 0     | 2        | kitten (0) | 3       | kitten (0) |
| 1     | 2        | kitten (0) | 4       | kitten (1) |
| 2     | 3        | kitten (1) | 3       | kitten (0) |
| 3     | 3        | kitten (1) | 4       | kitten (1) |
| 4     | 4        | kitten (2) | 3       | kitten (0) |
| 5     | 4        | kitten (2) | 4       | kitten (1) |
| 6     | 5        | zebra (0)  | 6       | zebra (0)  |

Joins are a backbone of data science and analytics. They're something we want in the new version and they're something we had in 1.0. But the high-level transforms allow for clean and efficient implementations without writing new kernels.

| index | key    | lower bound| upper bound |upper - lower| scan(upper - lower)|
|-------|--------|-------------|------------|-------------|--------------------|
|     0 | ape    | 0           | 0          | 0           | 0                  |
|     1 | ape    | 0           | 0          | 0           | 0                  |
|     2 | kitten | 3           | 5          | 2           | 0                  |
|     3 | kitten | 3           | 5          | 2           | 2                  |
|     4 | kitten | 3           | 5          | 2           | 4                  |
|     5 | zebra  | 6           | 7          | 1           | 6                  |
|       |        |             |            |             | 7                  |

First we call `sorted_search` twice: once to find the lower-bound and once to find the upper-bound of each element in A in B. This is a "find sorted needles in a sorted haystack" operation. The element-wise difference of the upper- and lower-bound indices gives the number of keys in B that match each key in A. We scan this difference to produce a segments-descriptor for load-balancing search, and use `transform_lbs` to construct an `int2` array with the (A index, B index) pairs of the inner join.

**`kernel_join.hxx`**
```cpp
template<typename launch_arg_t = empty_t, 
  typename a_it, typename b_it, typename comp_t>
mem_t<int2> inner_join(a_it a, int a_count, b_it b, int b_count, 
  comp_t comp, context_t& context) {

  // Compute lower and upper bounds of a into b.
  mem_t<int> lower(a_count, context);
  mem_t<int> upper(b_count, context);
  sorted_search<bounds_lower, launch_arg_t>(a, a_count, b, b_count, 
    lower.data(), comp, context);
  sorted_search<bounds_upper, launch_arg_t>(a, a_count, b, b_count, 
    upper.data(), comp, context);

  // Compute output ranges by scanning upper - lower. Retrieve the reduction
  // of the scan, which specifies the size of the output array to allocate.
  mem_t<int> scanned_sizes(a_count, context);
  const int* lower_data = lower.data();
  const int* upper_data = upper.data();

  mem_t<int> count(1, context);
  transform_scan([=]MGPU_DEVICE(int index) {
    return upper_data[index] - lower_data[index];
  }, a_count, scanned_sizes.data(), plus_t<int>(), count.data(), context);

  // Allocate an int2 output array and use load-balancing search to compute
  // the join.
  int join_count = from_mem(count)[0];
  mem_t<int2> output(join_count, context);
  int2* output_data = output.data();

  // Use load-balancing search on the segmens. The output is a pair with
  // a_index = seg and b_index = lower_data[seg] + rank.
  auto k = [=]MGPU_DEVICE(int index, int seg, int rank, tuple<int> lower) {
    output_data[index] = make_int2(seg, get<0>(lower) + rank);
  };
  transform_lbs<launch_arg_t>(k, join_count, scanned_sizes.data(), a_count,
    make_tuple(lower_data), context);

  return output;
}
```
The new `inner_join` implementation joins two tables as represented by sorted arrays. The arrays need not consist of arithmetic types; they only need to be comparable using the passed-in `comp` object, which returns `a < b`.

`sorted_search` takes a `bounds_lower` or `bounds_upper` argument for specializing on the search type. We then scan `a_count` sizes, which are element-wise differences of `upper` and `lower` as passed in through a lambda to the moderngpu `transform_scan` function. The reduction of this scan is returned into GPU-memory, and we load this count into a variable on the host where it's used for allocating just the right amount of memory for the join.

Once again `transform_lbs` provides the parallel intelligence for this operation. The indices for each pair of the inner join are:
```
a_index = seg;
b_index = lower_data[seg] + rank;
```
As in the `interval_move` example we choose to cache `lower_data[seg]` with the tuple-caching mechanism. 

This join implementation is easy to write, obvious on inspection, and robust with respect to the shape of the join. Running-time is linear in the sum of the two load-balancing search arguments: the join count and the size of table A.

### Breadth-first search

Features demonstrated:

1. `transform_scan`
2. `transform_lbs`

#### `bfs.cu`

```cpp
// Label vertices that have -1 value to cur_value + 1 if they are adjacent to a 
// vertex that is set tocur_value.
// Returns the size of the front for this pass.
template<typename vertices_it, typename edges_it>
int bfs(vertices_it vertices, int num_vertices, edges_it edges, int num_edges,
  int* values, int cur_value, context_t& context) {

  // Allocate space for load-balancing search segments and total work-items.
  mem_t<int> segments(num_vertices, context);
  mem_t<int> count(1, context);

  // Compute the number of neighbors to explore for each vertex.
  // Emit 0 items if the vertex is not active this pass, otherwise
  // emit vertices[vertex + 1] - vertices[vertex] items, indicating that we
  // need to visit all of the vertex's neighbors.

  // Scan the number of neighbors for each vertex on the frontier.
  auto segment_sizes = [=]MGPU_DEVICE(int vertex) {
    int count = 0;
    if(values[vertex] == cur_value) {
      int begin = vertices[vertex];
      int end = (vertex + 1 < num_vertices) ? vertices[vertex + 1] : num_edges;
      count = end - begin;
    }
    return count;
  };
  transform_scan(segment_sizes, num_vertices, segments.data(), plus_t<int>(), 
    count.data(), context);

  // Read out the number of work-items and quit if there are none. That means
  // we've visited all vertices connected to a source.
  int front = from_mem(count)[0];
  if(!front) return 0;

  // Compare-and-swap cur_value + 1 into values[edges[vertices[seg] + rank]]].
  // get<0>(v) is a cache-optimized copy of vertices[seg]. 
  auto update = [=]MGPU_DEVICE(int index, int seg, int rank, tuple<int> v) {
    // Compare and swap in cur_value + 1 when the old value was -1.
    int neighbor = edges[get<0>(v) + rank];
    atomicCAS(values + neighbor, -1, cur_value + 1);
  };
  transform_lbs(update, front, segments.data(), num_vertices, 
    make_tuple(vertices), context);

  return front;
}
```
```
Front for level 0 has 136 edges.
Front for level 1 has 19439 edges.
Front for level 2 has 5787 edges.
Front for level 3 has 92743 edges.
Front for level 4 has 543328 edges.
Front for level 5 has 4310689 edges.
Front for level 6 has 14915384 edges.
Front for level 7 has 9606212 edges.
Front for level 8 has 2005591 edges.
Front for level 9 has 408041 edges.
Front for level 10 has 125195 edges.
Front for level 11 has 25576 edges.
Front for level 12 has 7617 edges.
Front for level 13 has 5256 edges.
Front for level 14 has 1897 edges.
Front for level 15 has 437 edges.
Front for level 16 has 19 edges.
Front for level 17 has 18 edges.
Front for level 18 has 15 edges.
Front for level 19 has 12 edges.
Front for level 20 has 5 edges.
Front for level 21 has 10 edges.
Front for level 22 has 6 edges.
Front for level 23 has 11 edges.
Front for level 24 has 15 edges.
Front for level 25 has 1 edges.
Front for level 26 has 0 edges.
```
The moderngpu 2.0 demo "bfs" includes another practical algorithm has its load-balancing concerns satisfied by the high-level transforms. [Breadth-first search](https://en.wikipedia.org/wiki/Breadth-first_search) computes the minimum distance measured in edges for each vertex in a graph to one or more sources. This isn't an algorithmically efficient implementation--that would require clever management of the vertex frontier. But this is a load-balanced parallel implementation and it helps to demonstrate the facility of `transform_lbs`.

Breadth-first search updates an array of unweighted distances. Source vertices have distance 0. Visited vertices are marked by the number of edges to the closest source, and unvisited vertices are marked -1. This implementation of `bfs` updates the values array by marking unvisited vertices that are one edge away from vertices marked as `cur_value`.

Graphs are often encoded with compressed sparse row segment descriptors. The format used to encode sparse matrices helpfully share a format with the segment-descriptors arrays in moderngpu. Although we could feed this segments-descriptor array directly to the load-balancing search, we want to filter it to only process vertices on the frontier. We run `transform_scan` on the number of _active edges_ for each vertex at each step: 0 edges if the vertex _is not_ on the frontier at that step (i.e. `values[vertex] != cur_value`) or its full number of edges if the vertex _is_ on the frontier.

`transform_lbs` calls the provided lambda once for each edge attached to an active vertex. The high-level transform is robust with respect to the degree of the vertex and to the variance of the degree. We use the tuple segment caching mechanism to load the vertex's offset into the edge array (`tuple<int> v`), add in the rank of the edge within the vertex, and load from the `edge` array. This gives us the index of the neighboring vertex. The CUDA intrinsic `atomicCAS` sets the unweighted distance if and only if that vertex is unvisited.

A starting point for a more efficient BFS algorithm would be to use the return value of `atomicCAS` to make a list of vertices on the active front (to reduce or eliminate the cost of the `transform_scan` which operates over all vertices in the entire graph) or to remove some or all occurrences of visited vertices from the array of edges. moderngpu includes useful functions like `bulk_insert` and `bulk_remove` for the parallel array-surgery operations needed for more advanced bfs algorithms. 

### Attu Station, AK

Features demonstrated:

1. `scan`
2. `interval_expand`
3. `lbs_segreduce`
4. `segmented_sort` 

#### `cities.cu`

 ```
29510 cities from 52 states loaded.
AL          Phenix         Ladonia:   4.21    Smiths Stati:   7.12         Opelika:  24.78
AK    Attu Station            Adak: 435.04            Atka: 527.73        St. Paul: 712.48
AZ          Scenic     Littlefield:   6.58      Beaver Dam:   8.91    Grand Canyon:  55.75
AR        Junction       El Dorado:  13.95        Norphlet:  20.89          Strong:  21.88
CA         Needles       Bluewater:  48.57       Big River:  48.91          Blythe:  80.15
CO        Dinosaur         Rangely:  16.66         Maybell:  52.09          Meeker:  60.25
CT          Sharon       Lakeville:   5.79           Falls:   7.67          Canaan:  12.95
DE          Delmar          Laurel:   6.95          Bethel:   7.88          Blades:  11.67
DC      Washington
FL    Steinhatchee           Cross:  15.98    Horseshoe Be:  17.70    Fanning Spri:  27.97
GA           Fargo      Homerville:  26.30     Statenville:  26.62          Argyle:  27.09
HI        Maunaloa        Kualapuu:   9.65      Kaunakakai:  13.89        Ualapu'e:  24.68
ID          Salmon         Leadore:  43.09         Challis:  49.02         Clayton:  67.93
IL      Metropolis       Brookport:   3.75           Joppa:   9.71         Belknap:  18.32
IN    Mount Vernon    Parkers Sett:  12.30     New Harmony:  13.37      Poseyville:  17.20
IA       New Albin         Lansing:  10.03          Waukon:  18.45      Waterville:  19.95
KS        Kanorado        Goodland:  17.30     St. Francis:  32.57          Weskan:  32.77
KY          Albany     Burkesville:  14.43      Monticello:  18.84       Jamestown:  21.32
LA      Grand Isle    Golden Meado:  19.14        Galliano:  23.26    Port Sulphur:  27.35
ME         Houlton          Blaine:  26.14       Mars Hill:  27.54    Presque Isle:  39.94
MD      Manchester       Hampstead:   3.35     Westminster:   8.23     New Windsor:  13.64
MA      Siasconset       Nantucket:   6.42         Madaket:  11.12         Chatham:  28.24
MI        Newberry       St. James:  42.24      Manistique:  44.75      St. Ignace:  49.28
MN    Grand Marais          Lutsen:  17.21         Finland:  48.10      Silver Bay:  54.10
MS Alcorn State Un     Port Gibson:  10.63         Fayette:  12.42      Morgantown:  24.41
MO           Salem    Edgar Spring:  18.77         Licking:  20.31          Bunker:  21.72
MT          Jordan       Fort Peck:  51.91          Nashua:  61.74         Glasgow:  62.00
NE         Hyannis          Arthur:  29.83          Mullen:  36.97          Seneca:  47.89
NV          Owyhee    Fort McDermi:  73.38           Osino:  73.93    Paradise Val:  74.52
NH West Stewartsto       Colebrook:   6.84        Groveton:  26.20       Lancaster:  34.75
NJ     Port Norris     Laurel Lake:   5.28      Belleplain:   8.98       Millville:   9.67
NM         Clayton       Grenville:  27.74      Des Moines:  43.95          Folsom:  50.95
NY      Speculator     North Creek:  20.25       Long Lake:  26.53      Northville:  26.74
NC       Engelhard       Fairfield:  12.46    Swan Quarter:  18.68        Hatteras:  27.15
ND         Grenora         Fortuna:  21.27           Alamo:  21.55         Ambrose:  31.14
OH           Crown         Athalia:   5.60    Proctorville:  11.46      Chesapeake:  14.15
OK          Kenton            Felt:  25.51           Boise:  27.89           Keyes:  40.02
OR   Jordan Valley          Adrian:  52.62           Nyssa:  62.16          Harper:  67.03
PA       Driftwood        Emporium:  12.85    Prospect Par:  12.89       Pine Glen:  17.93
RI          Greene       Clayville:   4.82    Foster Cente:   5.93         Wyoming:  13.17
SC  McClellanville         Awendaw:  13.11       Jamestown:  19.00      Georgetown:  22.53
SD      Camp Crook         Buffalo:  21.02         Prairie:  56.03    Belle Fourch:  61.49
TN      Copperhill        Ducktown:   2.58          Benton:  19.10          Etowah:  24.69
TX        Van Horn    Sierra Blanc:  31.49       Valentine:  37.15    Fort Hancock:  62.19
UT        Wendover          Dugway:  74.97     Grantsville:  80.77     Rush Valley:  87.19
VT   Beecher Falls          Canaan:   1.93     Island Pond:  23.69      Derby Line:  30.30
VA        Monterey       Deerfield:  17.54     Craigsville:  24.99    Augusta Spri:  25.11
WA        Oroville          Loomis:  12.83        Tonasket:  16.40       Riverside:  30.16
WV      Brandywine        Franklin:   5.31         Whitmer:  21.26          Harman:  25.85
WI        Tomahawk     Rhinelander:  17.60         Merrill:  20.62    Lake Tomahaw:  23.72
WY         Mammoth            Alta:  84.55            Cody:  86.25         Ralston:  90.44
PR         Culebra         Vieques:  14.52       Esperanza:  18.53     Las Croabas:  21.85
 ```

The `cities.cu` demo implements a geographic data sciences query, not too far off from what a skilled SQL user might formulate. We start with a CSV file with American cities and their geodetic locations, sorted by US state. The goal is to find the three closest cities in each state for every city, as a sort of 'remoteness' metric. (3 is a template argument here. Any number will work but comparison costs increase quadratically.) The cities are then sorted by the distance to their third-closest city. This is a more robust measure of remoteness than taking simply the shortest pair. Two "census-designated places" may be close together yet very far from any other place. This measure is robust for up to three places in a cluster out by themselves--the third closest to each of those will be outside of the cluster and give truer sense of "remoteness." 

There are a limitless number of reasonable definitions of a remoteness measure: take the average distance to all other cities in the state, take the sum of the squared distances to all other cities, etc. The beauty of using moderngpu's high-level transformers is that most of the client code remains the same between these different metrics and only the reduction operator changes.

```cpp
template<int count>
struct best_t {
  struct term_t {
    float score;
    int index;
  } terms[count];

  best_t() = default;
  
  MGPU_HOST_DEVICE best_t(float score, int index) {
    fill(terms, term_t { FLT_MAX, -1 });
    terms[0] = term_t { score, index };
  }
};

struct combine_scores_t {
  template<int count>
  MGPU_HOST_DEVICE best_t<count> operator()(
    best_t<count> a, best_t<count> b) const {

    auto rotate = [](best_t<count> x) {
      best_t<count> y = x;
      iterate<count - 1>([&](int i) { y.terms[i] = y.terms[i + 1]; });
      return y;
    };
    best_t<count> c;

    iterate<count>([&](int i) {
      bool p = !(b.terms[0].score < a.terms[0].score);
      c.terms[i] = p ? a.terms[0] : b.terms[0];
      if(p) a = rotate(a); else b = rotate(b);
    });

    return c;
  }
};
```
Here we define the distance storage type `best_t<>`. If `count = 3`, it holds the indices of the three closest cities with their great-circle distances. The reduction operator chooses the three smallest distances from arguments `a` and `b`, each with three distances of their own. `combine_scores_t` is conscientiously designed to avoid dynamic indexing into any of the terms. We want to avoid spilling to high-latency local memory (dynamic indexing causes spilling into local memory), so we employ the compile-time loop-unwinding function `iterate<>`. We iteratively choose the smallest distance at the front of `a` or `b`, rotating forward the remaining distances of the respective argument. In theoretical terms this is an inefficient implementation, but rather than using memory operations it eats into register-register compute thoughput which the GPU has in abundance.

```cpp
template<int d>
std::unique_ptr<query_results<d> >
compute_distances(const int* cities_per_state, const float2* city_pos, 
  const state_city_t& db, context_t& context) {

  int num_states = (int)db.states.size();
  int num_cities = (int)db.cities.size();

  // 1: Scan the state counts for state segments.
  mem_t<int> state_segments(num_states, context);
  scan(cities_per_state, num_states, state_segments.data(), context);

  // 2: Use interval_expand to create a city->city map. This 
  mem_t<int> city_to_city_map(num_cities, context);
  interval_expand(state_segments.data(), num_cities, state_segments.data(),
    num_states, city_to_city_map.data(), context);

  // 3: For each city, store the number of cities in that city's state. 
  mem_t<int> distance_segments(num_cities, context);
  int* distance_segments_data = distance_segments.data();
  transform_lbs([=]MGPU_DEVICE(int index, int seg, int rank, 
    tuple<int> state_count) {
    distance_segments_data[index] = get<0>(state_count) - 1;
  }, num_cities, state_segments.data(), num_states, 
    make_tuple(cities_per_state), context);

  // 4: Scan the number of interactions for each state for the segment markers
  // for segmented reduction. Recover the number of work-items.
  scan(distance_segments.data(), num_cities, distance_segments.data(), context);

  // 5: Define a lambda that returns the minimum distance between two cities
  // in a best_t<best_count> struct. This is fed to lbs_segreduce to find the
  // best_count-closest cities within each state.

  // Compute the number of work-items. Each city performs distance tests on
  // cities_per_state[state] - 1 other cities. The -1 is to avoid computing
  // self-distances.
  int num_work_items = 0;
  for(int count : db.cities_per_state)
    num_work_items += count * (count - 1);

  // 6. Define a lambda that returns the distance between two cities.
  // Cache state_segments_data and pos_i.
  auto compute_score = [=]MGPU_DEVICE(int index, int city_i, int rank, 
    tuple<int, float2> desc) {

    // The position of city_i is cached.
    float2 pos_i = get<1>(desc);

    // The offset of the first city in this state is cached in desc.
    // Add in the rank to get city_i. 
    int city_j = get<0>(desc) + rank;

    // Advance city_j if city_j >= city_i to avoid self-interaction.
    if(city_j >= city_i) ++city_j;

    // Load the position of city_j and compute the distance to city_i.
    float2 pos_j = city_pos[city_j];
    float distance = earth_distance(pos_i.y, pos_i.x, pos_j.y, pos_j.x);

    // Set this as the closest distance in the structure.
    return best_t<d>(distance, city_j);
  };

  // Allocate on best_t<> per result.
  std::unique_ptr<query_results<d> > results(new query_results<d>);
  results->distances = mem_t<best_t<d> >(num_cities, context);
  results->indices = copy_to_mem(counting_iterator_t<int>(0), 
    num_cities, context);

  // 7. Call lbs_segreduce to fold all invocations of the 
  // Use fewer values per thread than the default lbs_segreduce tuning because
  // best_t<3> is a large type that eats up shared memory.
  best_t<d> init = best_t<d>();
  std::fill(init.terms, init.terms + d, typename best_t<d>::term_t { 0, -1 });

  lbs_segreduce<
    launch_params_t<128, 3>
  >(compute_score, num_work_items, distance_segments.data(), num_cities, 
    make_tuple(city_to_city_map.data(), city_pos), results->distances.data(), 
    combine_scores_t(), init, context);

  // 6: For each state, sort all cities by the distance of the (d-1`th) closest
  // city.  
  auto compare = []MGPU_DEVICE(best_t<d> left, best_t<d> right) {
    // Compare the least significant scores in each term.
    return left.terms[d - 1].score < right.terms[d - 1].score;
  };
  segmented_sort<
    launch_params_t<128, 3> 
  >(results->distances.data(), results->indices.data(), num_cities, 
    state_segments.data(), num_states, compare, context);
 
  return results;
}
```
`compute_distances` returns an array with `best_t<d>` structs for each city sorted in ascending distance of the (d-1'th) interval. This demo function uses several high-level transforms. `scan` and `interval_expand` are used together to turn the `cities_per_state` array (52 elements including District of Columbia and Puerto Rico) into a 29511-element map of cities intto the index of the first city in the same state, `city_to_city_map`. This provides a starting point inside the load-balancing search segmented reducer for computing the exact city-city distance pair of the work-item.

We utilize a brute-force all-pairs reducer. `transform_lbs` is used to record the number of distance calculations per city, `cities_per_state[state] - 1`, where the array element is loaded using tuple caching. The prefix sum of this generates `distance_segments`, the segments-descriptor array for the segmented reduction. We require two integers of storage per city. This is on the order of O(n) temporary memory, very storage-efficient compared to the O(n^2) number of distance calculations. 

The operative `lbs_segreduce` call requires only trivial storage space. Its usage here is as both a mapper and reducer: it expands work-items to compute all 25 million distance calculations, then reduces those results into 29 thousand closest-city tuples. GPUs are often memory capacity-limited devices, and the ability to perform on-the-fly expansions and reductions can be a make-or-break feature.

`lbs_segreduce` is a different, more fundamentally powerful function than the `transform_segreduce` used in [sparse matrix * vector](sparse-matrix--vector). The segment ID and rank-within-segment index are delivered to the user-provided lambda. In this query, the segment is the city and the rank is the index of the city within the state to compute the distance with. The rank is added to the iterator-cached `city_to_city_map` term to produce a global city index. The positions of each city are loaded (one from a cached iterator), the great-circle distance is computed, and a `best_t<>` structure is returned for reduction.

Finally, the moderngpu library function `segmented_sort` sorts each struct of distances and indices by ascending order within each state. The most-connected cities occur first, the most-remote cities last. The demo source writes all 29 thousand cities to a file and prints the most-remote cities to the console.

By our measure [Attu Station, AK](https://en.wikipedia.org/wiki/Attu_Station,_Alaska) is the most remote city in the United States. It's 435 miles from the nearest census-designated place and 712 miles from the third-nearest place. At the western extent of the Aleutian islands, Attu Station is actually in the eastern hemisphere and much closer to Russia than to the continental United States.

This demo is a starting point for all sorts of geographic data sciences queries. Moving from census data to Twitter, Instagram or Facebook data, we can imagine huge volumes of geo-tagged data available for accelerated analytics. By using moderngpu's high-level transforms, the business intelligence is kept inside the user's reduction operators; the library solves the parallel decomposition challenges of the GPU.

## Kernel programming
### Tutorial 1 - parallel transforms
**`tut_01_transform.cu`**
```cpp
#include <moderngpu/transform.hxx>      // for transform.
#include <cstdio>
#include <time.h>

int main(int argc, char** argv) {

  // Create an instance of an object that implements context_t.
  
  // context_t is an abstract base class that wraps basic CUDA runtime 
  // services like cudaMalloc and cudaFree.
  
  // standard_context_t is the trivial implementation of this abstract base
  // class. You can derive context_t and hook it up to your own memory
  // allocators, as CUDA's built-in allocator is very slow.

  mgpu::standard_context_t context;

  // Print the local time from GPU threads.
  time_t cur_time;
  time(&cur_time);
  tm t = *localtime(&cur_time);

  // Define a CUDA kernel with closure. Tag it with MGPU_DEVICE and compile
  // with --expt-extended-lambda in CUDA 7.5 to run it on the GPU.
  auto k = [=] MGPU_DEVICE(int index) {
    // This gets run on the GPU. Simply by referencing t.tm_year inside
    // the lambda, the time is copied from its enclosing scope on the host
    // into GPU constant memory and made available to the kernel.

    // Adjust for daylight savings.
    int hour = (t.tm_hour + (t.tm_isdst ? 0 : 11)) % 12;
    if(!hour) hour = 12;

    // Use CUDA's printf. It won't be shown until the context.synchronize()
    // is called.
    printf("Thread %d says the year is %d. The time is %d:%2d.\n", 
      index, 1900 + t.tm_year, hour, t.tm_min);
  };

  // Run kernel k with 10 GPU threads. We could even define the lambda 
  // inside the first argument of transform and not even name it.
  mgpu::transform(k, 10, context);

  // Synchronize the device to print the output.
  context.synchronize(); 

  return 0;
}
```
```
Thread 0 says the year is 2016. The time is 12:19.
Thread 1 says the year is 2016. The time is 12:19.
Thread 2 says the year is 2016. The time is 12:19.
Thread 3 says the year is 2016. The time is 12:19.
Thread 4 says the year is 2016. The time is 12:19.
Thread 5 says the year is 2016. The time is 12:19.
Thread 6 says the year is 2016. The time is 12:19.
Thread 7 says the year is 2016. The time is 12:19.
Thread 8 says the year is 2016. The time is 12:19.
Thread 9 says the year is 2016. The time is 12:19.
```
Launching threads on the GPU is easy. Simply define your kernel inside a C++ lambda function and use pass-by-value [=] closure to capture arguments from the enclosing scope. By tagging the lambda with `MGPU_DEVICE` (a macro for CUDA's `__device__` decorator) captured arguments are automatically copied from the host to the GPU's constant memory and made available to the kernel. 

The function argument for `mgpu::transform` may be any type that implements the method
```cpp
MGPU_DEVICE void operator()(int index);
```
C++ 11 supports specializing function templates over locally-defined types, so you may define the functor in function scope. However the language still prohibits defining templated local types, templated local functions, or templated methods of local types. Although less convenient, functor types with template methods are more flexible than lambda functions and may still be of utility in kernel development. These are supported by all the transform functions in moderngpu.

The `tm` struct captured in tut_01 is a plain-old-data type defined by the C runtime library. Types with non-trivial copy constructors or destructors may not be compatible with the extended-lambda capture mechanism supported by CUDA.

```cpp
struct context_t {
  context_t() = default;

  // Disable copy ctor and assignment operator. We don't want to let the
  // user copy only a slice.
  context_t(const context_t& rhs) = delete;
  context_t& operator=(const context_t& rhs) = delete;

  virtual int ptx_version() const = 0;
  virtual cudaStream_t stream() = 0;

  // Alloc GPU memory.
  virtual void* alloc(size_t size, memory_space_t space) = 0;
  virtual void free(void* p, memory_space_t space) = 0;

  // cudaStreamSynchronize or cudaDeviceSynchronize for stream 0.
  virtual void synchronize() = 0;

  virtual void timer_begin() = 0;
  virtual double timer_end() = 0;
};
```
`context_t` is an abstract base class through which CUDA runtime features may be accessed. `standard_context_t` implements this interface and provides the standard backend. `alloc` maps to `cudaMalloc` and `synchronize` maps to `cudaDeviceSynchronize`. Users concerned with performance should provide their own implementation of `context_t` which interfaces with the memory allocator used by the rest of their device code.

### Tutorial 2 - cooperative thread arrays
**`tut_02_cta_launch.cu`**
```cpp
template<int nt, typename input_it, typename output_it>
void simple_reduce(input_it input, output_it output, context_t& context) {
  typedef typename std::iterator_traits<input_it>::value_type type_t;

  // Use [=] to capture input and output pointers.
  auto k = [=] MGPU_DEVICE(int tid, int cta) {
    // Allocate shared memory and load the data in.
    __shared__ union { 
      type_t values[nt];
    } shared;
    type_t x = input[tid];
    shared.values[tid] = x;
    __syncthreads();

    // Make log2(nt) passes, each time adding elements from the right half
    // of the partial reductions into the left half of the partials.
    // At the end, everything is in shared.values[0].
    iterate<s_log2(nt)>([&](int pass) {
      int offset = (nt / 2)>> pass;
      if(tid < offset) shared.values[tid] = x += shared.values[tid + offset];
      __syncthreads();
    });

    if(!tid) *output = x;
  };  
  // Launch a grid for kernel k with one CTA of size nt.
  cta_launch<nt>(k, 1, context);
}
```
`cta_launch` and `cta_transform` launch _cooperative thread arrays_ (CTAs aka "thread blocks"). These functions are the preferred method for launching custom kernels with moderngpu. Unlike the lambdas called from `transform`, the lambdas invoked from `cta_launch` and `cta_transform` are passed both the local thread index `tid` (`threadIdx.x`) and the block index `cta` (`blockIdx.x`). Like `transform`, `cta_launch` may be called on either a functor object or a lambda function, but in either case a
```cpp
MGPU_DEVICE void operator()(int tid, int cta)
```
method must be provided.

`simple_reduce` is an example of what moderngpu 2.0 calls a _kernel_. The method exhibits cooperative parallelism by exchanging information through shared memory. Rather than a global index it uses a grid index `cta` and CTA-local thread ID `tid`. Additionally, the kernel captures more than just runtime parameters from its enclosing state: it uses the compile-time constant `nt` which controls the block size (the **n**umber of **t**hreads per CTA). This constant is shared between the device-side code, which uses it to size shared memory and control the number of loop iterations for the reduction, and the host-side `cta_launch` function, which needs the block size to pass to the launch chevron `<<<num_blocks, block_size>>>`.

### Tutorial 3 - launch box
**`tut_03_launch_box.cu`**
```cpp
int main(int argc, char** argv) {
  standard_context_t context;

  // One of these will be aliased to sm_ptx when the device code is compiled.
  typedef launch_box_t<
    arch_20_cta<128, 8>,    // Big Fermi GF100/GF110  eg GTX 580
    arch_21_cta<128, 4>,    // Lil Fermi GF10x/GF11x  eg GTX 550
    arch_30_cta<256, 4>,    // Lil Kepler GK10x       eg GTX 680
    arch_35_cta<256, 8>,    // Big Kepler GK110+      eg GTX 780 Ti
    arch_37_cta<256, 16>,   // Huge Kepler GK210      eg Tesla K80
    arch_50_cta<256, 8>,    // Lil Maxwell GM10x      eg GTX 750
    arch_52_cta<256, 16>    // Big Maxwell GM20x      eg GTX 980 Ti
  > launch_t;

  // We use [] to request no lambda closure. We aren't using values from
  // the surrounding scope--we're only using types.
  auto k = [] MGPU_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum { nt = params_t::nt, vt = params_t::vt };

    if(!tid) printf("Standard launch box: nt = %d vt = %d\n", nt, vt);
  };
  cta_launch<launch_t>(k, 1, context);

  context.synchronize();

  return 0;
}
```
On a GTX Titan (sm_35) device:
```
Standard launch box: nt = 256 vt = 8
```
The launch box mechanism in moderngpu 1.0 has been refactored and is more powerful and flexible than ever. One of the philosophies of the improved moderngpu is to push tuning out of the library and to the user. The library cannot anticipate every use, every type that its functions are specialized over, every operator used, or every length and distribution of data. It's not feasible to choose constants for each library function that provides the best performance for every variant. 

The user can specialize the variadic template class `launch_box_t` with a set of tuning parameters for each target architecture. The specialized types are typedef'd to symbols `sm_30`, `sm_35`, `sm_52` and so on. When the compiler generates device-side code, the `__CUDA_ARCH__` symbol is used to typedef `sm_ptx` to a a specific device architecture inside the launch box. The kernel definition (inside lambda `k` in this tutorial) extracts its parameters from the `sm_ptx` symbol of the launch box. Note that the launch box is not explicitly passed to the kernel, but accessed by name from the enclosing scope.

`arch_xx_cta` parameter structs derive the `launch_cta_t` primitive, which defines four parameters:
```cpp
template<int nt_, int vt_ = 1, int vt0_ = vt_, int occ_= 0>
struct launch_cta_t {
  enum { nt = nt_, vt = vt_, vt0 = vt0_, occ = occ_ };
};
```
`nt` controls the size of the CTA (thread block) in threads. `vt` is the number of values per thread (grain size). `vt0` is the number of unconditional loads of input made in cooperative parallel kernels (usually set to `vt` for regularly parallel functions and `vt - 1` for load-balancing search functions). `occ` is the occupancy argument of the `__launch_bounds__` kernel decorator. It specifies the minimum CTAs launched concurrently on each SM. 0 is the default, and allows the register allocator to optimize away spills by allocating many registers to hold live state. Setting a specific value for this increases occupancy by limiting register usage, but potentially causes spills to local memory. `arch_xx_cta` structs support all four arguments, although `nt` is the only argument required.

### Tutorial 4 - custom launch parameters

**`tut_04_launch_custom.cu`**
```cpp
enum modes_t {
  mode_basic = 1000,
  mode_enhanced = 2000, 
  mode_super = 3000
};

template<int nt_, modes_t mode_, typename type_t_> 
struct mode_param_t {
  // You must define nt, vt and occ (passed to __launch_bounds__) to use 
  // the launch box mechanism.
  enum { nt = nt_, vt = 1, vt0 = vt, occ = 0 };    // Required enums.
  enum { mode = mode_ };                           // Your custom enums.
  typedef type_t_ type_t;                          // Your custom types.
};

int main(int argc, char** argv) {
  standard_context_t context;

  // Define a launch box with a custom type. Use arch_xx to associate
  // a parameters type with a PTX version.
  typedef launch_box_t<
    arch_20<mode_param_t<64, mode_basic, short> >,    // HPC Fermi
    arch_35<mode_param_t<128, mode_enhanced, int> >,  // HPC Kepler 
    arch_52<mode_param_t<256, mode_super, int64_t> >  // HPC Maxwell
  > launch_custom_t;

  auto custom_k = [] MGPU_DEVICE(int tid, int cta) {
    typedef typename launch_custom_t::sm_ptx params_t;
    enum { nt = params_t::nt, mode = params_t::mode };
    typedef typename params_t::type_t type_t;

    if(!tid) 
      printf("Custom launch box: nt = %d mode = %d sizeof(type_t)=%d\n",
        nt, mode, sizeof(type_t));
  };
  cta_launch<launch_custom_t>(custom_k, 1, context);

  context.synchronize();

  return 0;
}
```
On a GTX Titan (sm_35) device:
```
Custom launch box: nt = 128 mode = 2000 sizeof(type_t)=4
```
Custom kernels may have different tuning parameters than the `nt`, `vt`, `vt0`, `occ` constants supported by the `arch_xx_cta` structs. The launch box mechanism supports the same architecture-specific selection of parameters from user-defined structs. Your custom structure can use any enum, typedef or constexpr. `arch_xx` is a template that serves to tag a user-defined type to a PTX code version. `nt`, `vt` and `occ` are still required to utilize the `cta_launch` and `cta_transform` functions, however the user can launch kernels without providing any of those constants by using the launch chevorn `<<<num_blocks, block_size>>>` directly and accessing version-specific structures directly through the `sm_ptx` tag.

### Tutorial 5 - iterators
**`tut_05_iterators.cu`**
```cpp
// Use Binet's Formula to compute the n'th Fibonacci number in constant
// time.
// http://www.cut-the-knot.org/proofs/BinetFormula.shtml  
MGPU_HOST_DEVICE int binet_formula(int index) {
  const double phi = 1.61803398875;
  int fib = (index < 2) ? index : 
    (int)((pow(phi, index) - pow(phi, -index)) / sqrt(5.0) + .5);
  return fib;
}

int main(int argc, char** argv) {
  standard_context_t context;

  // We call the function from a load iterator. Now it looks like we're
  // accessing an array.
  auto a = make_load_iterator<int>([]MGPU_DEVICE(int index) {
    return binet_formula(index);
  });

  // Print the first 20 Fibonacci numbers.
  transform([=]MGPU_DEVICE(int index) {
    int fib = a[index];
    printf("Fibonacci %2d = %4d\n", index, fib);
  }, 20, context);
  context.synchronize();

  // Merge with numbers starting at 50 and incrementing by 500 each time.
  strided_iterator_t<int> b(50, 500);

  int a_count = 20;
  int b_count = 10;

  // Make a store iterator that stores the logarithm of the merged value.
  mem_t<double2> c_device(a_count + b_count, context);
  double2* c_data = c_device.data();

  auto c = make_store_iterator<int>([=]MGPU_DEVICE(int value, int index) {
    c_data[index] = make_double2(value, value ? log((double)value) : 0);
  });

  // Merge iterators a and b into iterator c.
  merge(a, a_count, b, b_count, c, less_t<int>(), context);

  // Print the results.
  std::vector<double2> c_host = from_mem(c_device);
  for(double2 result : c_host)
    printf("log(%4d) = %f\n", (int)result.x, result.y);

  return 0;
}
```
moderngpu provides convenient iterators for passing arguments to array-processing functions without actually storing anything in memory. `counting_iterator_t` defines `operator[]` to return its index argument, which is useful for counting off the natural numbers in load-balancing search. `strided_iterator_t` scales the array-indexing argument and adds an offset. `make_store_iterator` and `make_load_iterator` adapt lambdas to iterators. These functions are a practical way of extending the functionality of any CUDA function that takes an iterator. This mechanism is used extensively in the implementation of moderngpu. For example, the array-processing prefix sum operator `scan` is defined to take an iterator for its input, but is extended into `transform_scan` by wrapping a user-provided lambda with `make_load_iterator`:
```cpp
template<scan_type_t scan_type = scan_type_exc, 
  typename launch_arg_t = empty_t, typename func_t, typename output_it,
  typename op_t, typename reduction_it>
void transform_scan(func_t f, int count, output_it output, op_t op,
  reduction_it reduction, context_t& context) {

  scan<scan_type, launch_arg_t>(make_load_iterator<decltype(f(0))>(f),
    count, output, op, reduction, context);
}
```
`scan` was written in the conventional manner (as in moderngpu 1.0) but extended to `transform_scan` with lambda iterators. Adapting lambdas to iterators can be both a convenience and a performance win.

## Usage tips

* Always compile with `-Xptxas="-v"`. This is a flag to the PTX assembler asking for verbose output. 
  ```
ptxas info    : Compiling entry function '_ZN4mgpu16launch_box_cta_kINS_15launch_params_tILi16ELi1ELi1ELi0EEEZ13simple_reduceILi16EPiS4_EvT0_T1_RNS_9context_tEEUnvdl2_PFvPiPiRN4mgpu9context_tEE13simple_reduceILi16EPiPiE1_PiPiEEvT0_' for 'sm_35'
ptxas info    : Function properties for _ZN4mgpu16launch_box_cta_kINS_15launch_params_tILi16ELi1ELi1ELi0EEEZ13simple_reduceILi16EPiS4_EvT0_T1_RNS_9context_tEEUnvdl2_PFvPiPiRN4mgpu9context_tEE13simple_reduceILi16EPiPiE1_PiPiEEvT0_
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, 64 bytes smem, 336 bytes cmem[0]
```
  This is the ptx verbose output for the intra-CTA reducer in the tutorial `tut_02_cta_launch.cu`. It only uses 8 registers and 64 bytes of shared memory. Occupancy is limited by max threads per SM (2048 on recent architectures), which means the user should increase grain size `vt` to do more work per thread. 

  If the PTX assembler's output lists any byte's spilled, it is likely that the kernel is attempting to dynamically index an array that was intended to sit in register. Only random-access memories like shared, local and device memory support random access. Registers must be accessed by name, so array accesses must be made with static indices, either hard-coded or produced from the indices of compile-time unrolled loops. CUDA has its own mechanism `#pragma unroll` for unrolling loops. Unfortunately this mechanism is just a hint, and the directive can be applied to many kinds of loops that do not unroll. Use moderngpu's `iterate<>` template to guarantee loop unrolling.

* Be careful with __host__ __device__-tagged functions.

  The CUDA compiler sometimes prohibits `__host__ __device__`-tagged code from calling `__device__`-tagged or `_`_host__`-tagged functions. However this restriction does not square with much of moderngpu's usage. We pass lambdas defined inside kernels (which are implicitly `__device__`-tagged) to the template loop-unrolling function `iterate<>`, which is `__host__ __device__`-tagged for greater generality. This compiles except when the enclosing scope of the `iterate<>` call is a non-template function. This is a dark area of the CUDA language where the forward-looking features have raced ahead of established features like tagging. If you are using moderngpu features and receive errors like 
  ```
  error: calling a __device__ function("operator()") from a __host__ __device__ function("eval") is not allowed
          detected during:
            instantiation of "void mgpu::iterate_t<i, count, valid>::eval(func_t) [with i=2, count=3, valid=true, func_t=lambda [](int)->void]"
  ```
  try making the enclosing scope a template function. There's no shame in tricking the compiler into doing what you want.
