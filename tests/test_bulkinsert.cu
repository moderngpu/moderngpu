#include <moderngpu/kernel_bulkinsert.hxx>

using namespace mgpu;

int main(int argc, char** argv) {

  standard_context_t context;

  // Insert a value like 10000+10i every 10th element.
  mem_t<int> output(1000 + 100, context);
  bulk_insert(strided_iterator_t<int>(10000, 10), 
    strided_iterator_t<int>(0, 10), 100, counting_iterator_t<int>(0), 1000,
    output.data(), context);

  std::vector<int> foo = from_mem(output);
  for(int i = 0; i < (int)foo.size(); ++i)
    printf("%4d: %6d\n", i, foo[i]);

  return 0;
}