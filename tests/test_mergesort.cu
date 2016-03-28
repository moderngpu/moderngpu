#include <moderngpu/kernel_mergesort.hxx>

using namespace mgpu;

int main(int argc, char** argv) {
  standard_context_t context;

  enum { nt = 128, vt = 11 };
  int count = 12345678;

  for(int it = 1; it <= 5; ++it) {

    mem_t<int> data = fill_random(0, 100000, count, false, context);

    mergesort(data.data(), count, less_t<int>(), context);

    std::vector<int> ref = from_mem(data);
    std::sort(ref.begin(), ref.end());
    std::vector<int> sorted = from_mem(data);

    bool print_sorted = ref != sorted;
    if(print_sorted) {
      for(int i = 0; i < div_up(count, vt); ++i) {
         printf("%4d: ", vt * i);
         for(int j = 0; j < vt; ++j)
           if(vt * i + j < count) printf("%5d ", sorted[vt * i + j]);
         printf("\n");
      }
    }
    
    printf("%3d %s\n", it, (ref == sorted) ? "SUCCESS" : "FAILURE");

    if(ref != sorted)
      return 0;
  }

  return 0;
}

