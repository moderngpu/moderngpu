#include <moderngpu/kernel_join.hxx>

using namespace mgpu;

int main(int argc, char** argv) {
  standard_context_t context;
  mem_t<int2> joined = inner_join((int*)nullptr, 0, (int*)nullptr, 0,
    less_t<int>(), context);
  return 0;
}
