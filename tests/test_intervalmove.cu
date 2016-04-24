#include <moderngpu/kernel_intervalmove.hxx>

using namespace mgpu;

int main(int argc, char** argv) {
  standard_context_t context;
  
  int b_count = 64;
  int a_count = 512 - b_count;
  std::vector<int> segs_host(b_count);
  for(int i = 0; i < b_count; ++i)
    segs_host[i] = 5 * i;
  mem_t<int> segs = to_mem(segs_host, context);

  interval_move((const int*)0, a_count, segs.data(), b_count, 
    (const int*)0, (const int*)0, (int*)0, context);

  cudaDeviceSynchronize();

  return 0;

}
