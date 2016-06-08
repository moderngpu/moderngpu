#include <moderngpu/kernel_merge.hxx>

using namespace mgpu;

int main(int argc, char** argv) {
  standard_context_t context;

  // Loop from 1K to 100M.
  for(int count = 1000; count <= 100000000; count += count / 10) {
    int a_count = count / 2;
    int b_count = count - a_count;

    mem_t<int> a = fill_random(0, count, a_count, true, context);
    mem_t<int> b = fill_random(0, count, b_count, true, context);
    mem_t<int> c(count, context);

    merge(a.data(), a_count, b.data(), b_count, c.data(), 
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
    printf("%8d: %s\n", count, success ? "SUCCESS" : "FAILURE");
    if(!success) exit(1);
  }

  return 0;
}

