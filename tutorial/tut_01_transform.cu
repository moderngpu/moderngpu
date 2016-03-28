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