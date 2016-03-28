#include <moderngpu/kernel_segsort.hxx>

using namespace mgpu;

std::vector<int> cpu_segsort(const std::vector<int>& data,
  const std::vector<int>& segments) {

  std::vector<int> copy = data;
  int cur = 0;
  for(int seg = 0; seg < segments.size(); ++seg) {
    int next = segments[seg];
    std::sort(copy.data() + cur, copy.data() + next);
    cur = next;
  }
  std::sort(copy.data() + cur, copy.data() + data.size());
  return data;
}

int main(int argc, char** argv) {
  standard_context_t context;

  for(int count = 100; count < 23456789; count += count / 100) {

    for(int it = 1; it <= 100; ++it) {

      int num_segments = div_up(count, 30);
      mem_t<int> segs = fill_random(0, count - 1, num_segments, true, context);
      std::vector<int> segs_host = from_mem(segs);
      mem_t<int> data = fill_random(0, 100000, count, false, context);

      segmented_sort<
        launch_params_t<256, 5>
      >(data.data(), count, segs.data(), num_segments, 
        less_t<int>(), context);

      std::vector<int> ref = cpu_segsort(from_mem(data), segs_host);
      std::vector<int> sorted = from_mem(data);

      bool print_sorted = ref != sorted;
      if(print_sorted) {
        enum { width = 4 };
        for(int i = 0; i < div_up(count, width); ++i) {
           printf("%4d: ", width * i);
           for(int j = 0; j < width; ++j) {
              int index = width * i + j;
             if(index < count) printf("%5d ", sorted[index]);
           }
           printf("\n");
        }
      }
     
      printf("count = %8d it = %3d %s\n", count, it, 
        (ref == sorted) ? "SUCCESS" : "FAILURE");

      if(ref != sorted)
        return 0;
    }
  }

  return 0;
}

