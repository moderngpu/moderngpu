
# moderngpu 2.0 
**(c) 2016 [Sean Baxter](http://twitter.com/moderngpu)** 

**You can drop me a line [here](mailto:moderngpu@gmail.com)**


**Latest update**:
```
2.06 2016 Apr 12 - 
  Fixed critical kernel versioning bug. Now uses cudaFuncGetAttributes
  ptxVersion instead of binaryVersion to select launch box parameters.
  Thanks @antonmks.
```
---
moderngpu is a productivity library for general-purpose computing on GPUs. It is a header-only C++ library written for CUDA. The unique value of the library is in its accelerated primitives for solving irregularly parallel problems. 

## Contents
1. [Release notes](#release-notes)
  1. [License](LICENSE.md)
1. [Introduction](LOADBALANCE.md#introduction)
1. [Examples of automatic load-balancing](LOADBALANCE.md#examples-of-automatic-load-balancing)
  1. [Sparse matrix*vector](LOADBALANCE.md#sparse-matrix--vector)
  1. [Interval move and interval expand](LOADBALANCE.md#interval-move-and-interval-expand)
  1. [Relational join](LOADBALANCE.md#relational-join)
  1. [Breadth-first search](LOADBALANCE.md#breadth-first-search)
  1. [Attu Station, AK](LOADBALANCE.md#attu-station-ak)
1. [Examples of dynamic work creation](LOADBALANCE.md#examples-of-dynamic-work-creation)
  1. [Improved breadth-first search](LOADBALANCE.md#improved-breadth-first-search)
  1. [Bit-compressed breadth-first search](LOADBALANCE.md#bit-compressed-breadth-first-search)
1. [Kernel programming](KERNEL.md#kernel-programming)
  1. [Tutorial 1 - parallel transforms](KERNEL.md#tutorial-1---parallel-transforms)
  1. [Tutorial 2 - cooperative thread arrays](KERNEL.md#tutorial-2---cooperative-thread-arrays)
  1. [Tutorial 3 - launch box](KERNEL.md#tutorial-3---launch-box)
  1. [Tutorial 4 - custom launch parameters](KERNEL.md#tutorial-4---custom-launch-parameters)
  1. [Tutorial 5 - iterators](KERNEL.md#tutorial-5---iterators)
1. [API Reference](API.md)
  1. [Low-level transforms](API.md#low-level-transforms)
  1. [High-level transforms](API.md#high-level-transforms)
  1. [Array functions](API.md#array-functions)
1. [Usage tips](USAGE.md#usage-tips)
1. [ModernGPU 1.0](https://github.com/moderngpu/moderngpu/tree/V1.1)
  1. [FAQ](https://moderngpu.github.io/moderngpu/faq.html)
  1. [Introduction](https://moderngpu.github.io/moderngpu/intro.html)
  1. [Performance](https://moderngpu.github.io/moderngpu/performance.html)
  1. [The Library](https://moderngpu.github.io/moderngpu/library.html)
  1. [Reduce and Scan](https://moderngpu.github.io/moderngpu/scan.html)
  1. [Bulk Remove and Bulk Insert](https://moderngpu.github.io/moderngpu/bulkinsert.html)
  1. [Merge](https://moderngpu.github.io/moderngpu/merge.html)
  1. [Mergesort](https://moderngpu.github.io/moderngpu/mergesort.html)
  1. [Segmented Sort and Locality Sort](https://moderngpu.github.io/moderngpu/segsort.html)
  1. [Vectorized Sorted Search](https://moderngpu.github.io/moderngpu/sortedsearch.html)
  1. [Load-Balancing Search](https://moderngpu.github.io/moderngpu/loadbalance.html)
  1. [IntervalExpand and IntervalMove](https://moderngpu.github.io/moderngpu/intervalmove.html)
  1. [Relational Joins](https://moderngpu.github.io/moderngpu/join.html)
  1. [Multisets](https://moderngpu.github.io/moderngpu/sets.html)
  1. [Segmented Reduction](https://moderngpu.github.io/moderngpu/segreduce.html)

## Release notes
```
2.06 2016 Apr 12 - 
  Fixed critical kernel versioning bug. Now uses cudaFuncGetAttributes
  ptxVersion instead of binaryVersion to select launch box parameters.
  Thanks @antonmks.
  
2.05 2016 Apr 3 -
  Restructured segmented sort and added segmented_sort_indices.
  Wrote more robust test_segsort.cu test.
  Modified cities demo to use segmented_sort_indices.
  TODO: Segmented sort with bitfield for segmented headers.

2.04 2016 Apr 2 -
  Fixed multiple definition of dummy_k kernel when including 
    standard_context_t in multiple translation units.

2.03 2016 Apr 2 -
  Added transform_compact pattern and test_compact.cu.

2.02 2016 Apr 1 -
  Added dynamic work-creation function lbs_workcreate.
  Improved ease of calling cta_scan_t.
  cta_reduce_t now uses shfl for all datatypes and operators on sm_30+.

2.01 2016 Mar 31 -
  Refactored scan implementation and added scan_event.

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
