#include <moderngpu/kernel_compact.hxx>

using namespace mgpu;

int main(int argc, char** argv) {

  standard_context_t context;

  for(int count = 100; count <= 23456789; count += count / 20) {
    // Construct the compaction state with transform_compact.
    auto compact = transform_compact(count, context);

    // The upsweep determines which items to compact.
    int stream_count = compact.upsweep([]MGPU_DEVICE(int index) {
      // Return true if we want to keep this element.
      // Only stream if the number of bits in index is a multiple of 5.
      bool keep = 0 == (popc(index) % 5);
      return keep;
    });

    // Compact the results into this buffer.
    mem_t<int> special_numbers(stream_count, context);
    int* special_numbers_data = special_numbers.data();
    compact.downsweep([=]MGPU_DEVICE(int dest_index, int source_index) {
      special_numbers_data[dest_index] = source_index;
    });

    // Test the results.
    std::vector<int> host = from_mem(special_numbers);
    int j = 0;
    for(int i = 0; i < count; ++i) {
      if(0 == (popc(i) % 5)) {
        // i had better be in special_numbers.
        if(host[j] != i) {
          printf("Streaming error at count = %d, i = %d\n", count, i);
          exit(0);
        }
        ++j;
      }
    }
    if(j != stream_count) {
      printf("Streaming error at count = %d. Wrong size.\n", count);
      exit(0);
    }

    printf("count %d stream_count = %d\n", count, stream_count);
  }

  return 0;
}
