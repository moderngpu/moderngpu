#include <moderngpu/kernel_bulkremove.hxx>

using namespace mgpu;

int main(int argc, char** argv) {
  standard_context_t context;

  int count = 10000;
  int remove = 1000;
  mem_t<int> output(count - remove, context);
  bulk_remove(counting_iterator_t<int>(0), count, 
    strided_iterator_t<int>(5, 10), remove, output.data(), context);

  std::vector<int> output_host = from_mem(output);

  for(int i = 0; i < (int)output_host.size(); ++i) {
    printf("%4d: %4d\n", i, output_host[i]);
  }

  return 0;
}
