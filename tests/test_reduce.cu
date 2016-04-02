#include <moderngpu/kernel_reduce.hxx>
#include <moderngpu/memory.hxx>
#include <numeric> // std:accumulate

using namespace mgpu;

int main(int argc, char** argv) {

  standard_context_t context;

  for(int count = 1000; count < 23456789; count += count / 100) {
    mem_t<int> input = fill_random(0, 100, count, false, context);
    const int* input_data = input.data();

    mem_t<int> reduction(1, context);

    // reduce()
    reduce(input_data, count, reduction.data(), plus_t<int>(), context);
    context.synchronize();
    std::vector<int> result1 = from_mem(reduction);

    // transform_reduce()
    // construct a lambda that returns input_data[index].
    auto f = [=]MGPU_DEVICE(int index) { return input_data[index]; };
    //transform_reduce(f, count, reduction.data(), plus_t<int>(), context);
    std::vector<int> result2 = from_mem(reduction);

    // host reduce using std::accumulate.
    std::vector<int> input_host = from_mem(input);
    int ref = std::accumulate(input_host.begin(), input_host.end(), 0);

    if(result1[0] != ref || result2[0] != ref) {
      printf("reduce:           %d\n", result1[0]);
      printf("transform_reduce: %d\n", result2[0]);
      printf("std::accumulate:  %d\n", ref);
      printf("ERROR AT COUNT = %d\n", count);
      exit(0);
    } else
      printf("Reduction for count %d success\n", count);
  }
  return 0; 

}