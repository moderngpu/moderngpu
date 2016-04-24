#include <moderngpu/transform.hxx>
#include <moderngpu/kernel_segreduce.hxx>

using namespace mgpu;


template<typename func_t, typename... args_t>
__global__ void k(func_t f, args_t... args) {
  f(threadIdx.x, blockIdx.x, make_restrict(args)...);
}

template<typename func_t, typename... args_t>
void simple_transform(func_t f, args_t... args) { 
  k<<<1, 128>>>(f, args...);
}

int main(int argc, const char**) {

  standard_context_t context;

  const int* input = nullptr;
  int* output = nullptr;
  int count = 100000;
  
  transform<128, 8>([=]MGPU_DEVICE(int index, const int* input, int* output) {
    output[index] = 2 * input[index];
  }, count, context, input, output);

  spmv((const double*)nullptr, (const int*)nullptr, (const double*)nullptr, 
    0, (const int*)nullptr, 0, (double*)nullptr, context);

  context.synchronize();

  return 0;
}