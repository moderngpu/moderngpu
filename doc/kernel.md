1. [moderngpu 2.0...](README.md)
1. [Kernel programming](#kernel-programming)
  1. [Tutorial 1 - parallel transforms](#tutorial-1---parallel-transforms)
  1. [Tutorial 2 - cooperative thread arrays](#tutorial-2---cooperative-thread-arrays)
  1. [Tutorial 3 - launch box](#tutorial-3---launch-box)
  1. [Tutorial 4 - custom launch parameters](#tutorial-4---custom-launch-parameters)
  1. [Tutorial 5 - iterators](#tutorial-5---iterators)

# Kernel programming

## Tutorial 1 - parallel transforms
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

## Tutorial 2 - cooperative thread arrays
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

## Tutorial 3 - launch box
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

## Tutorial 4 - custom launch parameters

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

## Tutorial 5 - iterators
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
```
Fibonacci  0 =    0
Fibonacci  1 =    1
Fibonacci  2 =    1
Fibonacci  3 =    2
Fibonacci  4 =    3
Fibonacci  5 =    5
Fibonacci  6 =    8
Fibonacci  7 =   13
Fibonacci  8 =   21
Fibonacci  9 =   34
Fibonacci 10 =   55
Fibonacci 11 =   89
Fibonacci 12 =  144
Fibonacci 13 =  233
Fibonacci 14 =  377
Fibonacci 15 =  610
Fibonacci 16 =  987
Fibonacci 17 = 1597
Fibonacci 18 = 2584
Fibonacci 19 = 4181
log(   0) = 0.000000
log(   1) = 0.000000
log(   1) = 0.000000
log(   2) = 0.693147
log(   3) = 1.098612
log(   5) = 1.609438
log(   8) = 2.079442
log(  13) = 2.564949
log(  21) = 3.044522
log(  34) = 3.526361
log(  50) = 3.912023
log(  55) = 4.007333
log(  89) = 4.488636
log( 144) = 4.969813
log( 233) = 5.451038
log( 377) = 5.932245
log( 550) = 6.309918
log( 610) = 6.413459
log( 987) = 6.894670
log(1050) = 6.956545
log(1550) = 7.346010
log(1597) = 7.375882
log(2050) = 7.625595
log(2550) = 7.843849
log(2584) = 7.857094
log(3050) = 8.022897
log(3550) = 8.174703
log(4050) = 8.306472
log(4181) = 8.338306
log(4550) = 8.422883
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
