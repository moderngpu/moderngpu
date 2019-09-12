
# moderngpu 2.0 
**(c) 2016 [Sean Baxter](http://twitter.com/moderngpu)** 

**You can drop me a line [here](mailto:moderngpu@gmail.com)**

Full documentation with [github wiki](https://github.com/moderngpu/moderngpu/wiki) under heavy construction.

**Latest update**:
```
2.12 2016 June 8 -
  Fixed problem in load_two_streams_reg when loading from unaligned types.
```
---
moderngpu is a productivity library for general-purpose computing on GPUs. It is a header-only C++ library written for CUDA. The unique value of the library is in its accelerated primitives for solving irregularly parallel problems. 

## Release notes
```
2.11 2016 June 6 -
  Removed decltype() calls on __device__-tagged lambdas. This introduces
    two breaking changes: transform_scan and fill_function now take explicit
    types as their first template arguments.

2.10 2016 May 15 -
  Allow for non-pow2 sized launches. Rewrote cta_reduce_t to support these
  sizes.

2.09 2016 May 7 -
  Greatly improved and more standard tuple class.
  Optimized tuple value caching for lbs-related functions. 

2.08 2016 Apr 24 -
  Restricted pointer promotion on transform functions using variadic arguments.
  Fixed reduction bug in stream compaction.

2.07 2016 Apr 17 -
  Added mechanism for passing kernel arguments through variadic parameter pack.
  Added occupancy calculator in launch_box.hxx.
  
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
  
  Developed and tested with CUDA 7.5.17, g++ 4.9.3 on 64-bit linux.

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
