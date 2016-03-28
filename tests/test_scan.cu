#include <moderngpu/kernel_scan.hxx>
#include <iostream>

using namespace mgpu;

int main(int argc, char** argv) {
  standard_context_t context;

  typedef double type_t;
  int count = 50000;
  
  mem_t<type_t> output(count, context);
  mem_t<type_t> total(1, context, memory_space_host);

  scan<scan_type_exc>(constant_iterator_t<type_t>(1.01), count, output.data(), 
    plus_t<type_t>(), total.data(), context);

  cudaStreamSynchronize(0);
  printf("TOTAL = %f\n", total.data()[0]);

  std::vector<type_t> y = from_mem(output);

  for(int i = 0; i < count; ++i)
    printf("%3d %f\n", i, y[i]);

  return 0;
}
