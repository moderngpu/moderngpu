#include <moderngpu/transform.hxx>   // for cta_launch.
#include <moderngpu/memory.hxx>      // for mem_t.
#include <cstdio>

using namespace mgpu;

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

int main(int argc, char** argv) {

  standard_context_t context;

  // Perform an intra-cta reduction on the first 16 perfect squares.
  // The choice of 16 has structural implications (number of threads per CTA)
  // so we make it a compile-time constant.
  enum { nt = 16 };

  // Prepare the fibonacci numbers on the host.
  std::vector<int> input_host(nt);
  for(int i = 0; i < nt; ++i)
    input_host[i] = (i + 1) * (i + 1);

  printf("Reducing: ");
  for(int i : input_host) printf("%d ", i);
  printf("...\n");

  // Copy the data to the GPU.
  mem_t<int> input_device = to_mem(input_host, context);

  // Call our simple reduce.
  mem_t<int> output_device(1, context);
  simple_reduce<nt>(input_device.data(), output_device.data(), context);

  // Print the reduction.
  std::vector<int> output_host = from_mem(output_device);
  printf("Reduction of %d elements is %d\n", nt, output_host[0]);

  return 0;
}