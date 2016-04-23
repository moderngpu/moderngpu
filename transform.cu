#include <moderngpu/transform.hxx>

using namespace mgpu;

template<typename func_t, typename... args_t>
__global__ void k(func_t f, args_t... args) {
  auto foo = make_tuple(args...);
  auto restricted = restrict_tuple(foo);
//  const int* __restrict__ input = get<0>(restricted);
//  int* __restrict__ output = get<1>(restricted);

  f(threadIdx.x, blockIdx.x, get<0>(restricted), get<1>(restricted));
}

template<typename func_t, typename... args_t>
void simple_transform(func_t f, args_t... args) { 
  k<<<1, 128>>>(f, args...);
}

int main(int argc, const char**) {
  typedef tuple<const float*, int, double, char*, const short*> tpl_t;
  tpl_t unrestriced = tpl_t();

  auto restricted = restrict_tuple(unrestriced);

  auto x0 = get<0>(restricted);
  auto x1 = get<1>(restricted);
  auto x2 = get<2>(restricted);
  auto x3 = get<3>(restricted);
  auto x4 = get<4>(restricted);



  printf("%d %d %d %d %d\n",
    is_restrict<decltype(get<0>(restricted))>::value,
    is_restrict<decltype(get<1>(restricted))>::value,
    is_restrict<decltype(get<2>(restricted))>::value,
    is_restrict<decltype(get<3>(restricted))>::value,
    is_restrict<decltype(get<4>(restricted))>::value
  );
/*  standard_context_t context;

  const int* input = nullptr;
  int* output = nullptr;
  int count = 100000;

  auto f = [=]MGPU_DEVICE(int tid, int cta, const int* input, int* output) {

    output[128 * 0 + tid] = 2 * input[128 * 0 + tid];
    output[128 * 1 + tid] = 2 * input[128 * 1 + tid];
    output[128 * 2 + tid] = 2 * input[128 * 2 + tid];
    output[128 * 3 + tid] = 2 * input[128 * 3 + tid];
    output[128 * 4 + tid] = 2 * input[128 * 4 + tid];
  };
  simple_transform(f, input, output);



//  transform<128, 8>([=]MGPU_DEVICE(int index, const int* input, int* output) {
//    output[index] = 2 * input[index];
//  }, count, context, input, output);

  cta_launch<128, 8>([=]MGPU_DEVICE(int tid, int cta, 
    const int* __restrict__ input, int* __restrict__ output) {
    output[128 * 0 + tid] = 2 * input[128 * 0 + tid];
    output[128 * 1 + tid] = 2 * input[128 * 1 + tid];
    output[128 * 2 + tid] = 2 * input[128 * 2 + tid];
    output[128 * 3 + tid] = 2 * input[128 * 3 + tid];
    output[128 * 4 + tid] = 2 * input[128 * 4 + tid];
  }, 1, context, input, output);
*/


  return 0;
}