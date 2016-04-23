#include <moderngpu/transform.hxx>

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

  auto f = [=]MGPU_DEVICE(int tid, int cta, const int* input, int* output) {
    if(input && output) {
      output[128 * 0 + tid] = 2 * input[128 * 0 + tid];
      output[128 * 1 + tid] = 2 * input[128 * 1 + tid];
      output[128 * 2 + tid] = 2 * input[128 * 2 + tid];
      output[128 * 3 + tid] = 2 * input[128 * 3 + tid];
      output[128 * 4 + tid] = 2 * input[128 * 4 + tid];
      output[128 * 5 + tid] = 2 * input[128 * 5 + tid];
      output[128 * 6 + tid] = 2 * input[128 * 6 + tid];
      output[128 * 7 + tid] = 2 * input[128 * 7 + tid];
    }
  };
  simple_transform(f, input, output);

  cta_launch<128, 8>([=]MGPU_DEVICE(int tid, int cta, const int* input, int* output) {
    if(input && output) {
      output[128 * 0 + tid] = 2 * input[128 * 0 + tid];
      output[128 * 1 + tid] = 2 * input[128 * 1 + tid];
      output[128 * 2 + tid] = 2 * input[128 * 2 + tid];
      output[128 * 3 + tid] = 2 * input[128 * 3 + tid];
      output[128 * 4 + tid] = 2 * input[128 * 4 + tid];
      output[128 * 5 + tid] = 2 * input[128 * 5 + tid];
      output[128 * 6 + tid] = 2 * input[128 * 6 + tid];
      output[128 * 7 + tid] = 2 * input[128 * 7 + tid];
    }
  }, 1, context, input, output);

  transform<128, 8>([=]MGPU_DEVICE(int index, const int* input, int* output) {
    output[index] = 2 * input[index];
  }, count, context, input, output);

  context.synchronize();

  return 0;
}