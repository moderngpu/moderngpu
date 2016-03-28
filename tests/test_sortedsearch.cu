#include <moderngpu/kernel_sortedsearch.hxx>

using namespace mgpu;

int main(int argc, char** argv) {
  standard_context_t context;

  int num_needles = 1000;
  int num_haystack = 1000;
  mem_t<int> needles = fill_random(0, 10000, num_needles, true, context);
  mem_t<int> haystack = fill_random(0, 10000, num_haystack, true, context);

  mem_t<int> indices(num_needles, context);
  sorted_search<bounds_lower>((int*)needles.data(), num_needles, 
    (int*)haystack.data(),
    num_haystack, indices.data(), less_t<int>(), context);

  std::vector<int> needles_host = from_mem(needles);
  std::vector<int> haystack_host = from_mem(haystack);
  std::vector<int> indices_host = from_mem(indices);

  for(int i = 0; i < (int)indices_host.size(); ++i) {
    int needle = needles_host[i];
    int index = indices_host[i];
    
    assert(index <= 0 || needle > haystack_host[index - 1]);
    assert(index >= num_haystack || needle <= haystack_host[index]);

  }

  return 0;  
}