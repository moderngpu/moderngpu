#include <moderngpu/kernel_load_balance.hxx>

using namespace mgpu;

int main(int argc, char** argv) {

  standard_context_t context;

  int count = 200030;
  int spacing = 100;

  int num_segments = div_up(count, spacing);
  std::vector<int> segments_host(num_segments);
  for(int i = 0; i < num_segments; ++i)
    segments_host[i] = i * spacing;
  mem_t<int> segments = to_mem(segments_host, context);

  mem_t<int> lbs(count, context);
  load_balance_search(count, segments.data(), num_segments, lbs.data(), 
    context);

  std::vector<int> lbs_host = from_mem(lbs);
  for(size_t i = 0; i < lbs_host.size(); ++i) {
    printf("%4d: %3d\n", (int)i, lbs_host[i]);
    if(lbs_host[i] != i / spacing) {
      printf("ERROR AT %d\n", (int)i);
      exit(0);
    }
  }

  return 0;
}
