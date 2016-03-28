#include <moderngpu/kernel_merge.hxx>

using namespace mgpu;

int main(int argc, char** argv) {
  standard_context_t context;

  int a_count = 1234567;
  int b_count = 1234567;
  int count = a_count + b_count;

  mem_t<int> a = fill_random(0, 10000, a_count, true, context);
  mem_t<int> b = fill_random(0, 10000, b_count, true, context);
  mem_t<int> c(count, context);

  typedef launch_params_t<128, 7> launch_t;

  merge<launch_t>(a.data(), a.size(), b.data(), b.size(), c.data(), 
    mgpu::less_t<int>(), context);

  // Download the results.
  std::vector<int> a_host = from_mem(a);
  std::vector<int> b_host = from_mem(b);
  std::vector<int> c_host = from_mem(c);

  // Do merge on the host and compare.
  std::vector<int> c2(count);
  std::merge(a_host.begin(), a_host.end(), b_host.begin(), b_host.end(),
    c2.begin());

  bool success = c2 == c_host;
  printf(success ? "SUCCESS\n" : "FAILURE\n");

  return 0;
}

