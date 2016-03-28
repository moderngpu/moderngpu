#include <moderngpu/transform.hxx>    // cta_transform
#include <moderngpu/memory.hxx>       // fill_random
#include <moderngpu/cta_merge.hxx>    // merge_path
#include <moderngpu/operators.hxx>    // less_t

using namespace mgpu;

template<typename launch_t, typename a_it, typename b_it, typename c_it, 
  typename comp_t>
void simple_merge(a_it a, int a_count, b_it b, int b_count, c_it c,
  comp_t comp, context_t& context) {

  typedef typename std::iterator_traits<c_it>::value_type type_t;

  int count = a_count + b_count;
  auto k = [=] MGPU_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };
    // We really should buffer the inputs and outputs in shared memory
    // with a union like this:
    // __shared__ union {
    //   type_t values[nv];
    // } shared;

    // Get the position of the current tile. This returns the interval
    // [nv * cta, min(count, nv * (cta + 1))).
    range_t tile = get_tile(cta, nv, count);

    // Find the starting point to merge from a and b for this thread.
    int diag = tile.begin + vt * tid;
    int mp = merge_path(a, a_count, b, b_count, diag, comp);
    int a_index = mp;
    int b_index = diag - mp;

    // Sequentially merge up to vt elements per thread.
    // thread_iterate calls its argument up to vt times per thread for
    // j in the range [0, count).
    // Use [&] closure to update the counters a_index and b_index.
    thread_iterate<nt, vt>(tid, tile.count(), [&](int i, int j) {
      bool p;
      if(a_index >= a_count) p = false;
      else if(b_index >= b_count) p = true;
      else p = !comp(b[b_index], a[a_index]);   // !(b < a) == (a <= b).

      type_t output = p ? a[a_index++] : b[b_index++];
      c[tile.begin + j] = output;
    });
  };

  // cta_transform launches enough CTAs to process count elements with 
  // nt * vt elements per CTA. It always calls the kernel lambda with the 
  // complete number of threads per CTA. That is, it calls the lambda with
  // complete, not partial, CTAs.
  cta_transform<launch_t>(k, count, context);
}

int main(int argc, char** argv) {
  standard_context_t context;

  int a_count = 50000;
  int b_count = 50000;
  int count = a_count + b_count;

  // Allocate and fill arrays with sorted random integers between 0 and count.
  mem_t<int> a_device = fill_random(0, count, a_count, true, context);
  mem_t<int> b_device = fill_random(0, count, b_count, true, context);
  mem_t<int> c_device(count, context);

  // Launch blocks of 128 threads and assign 8 values to each thread.
  // It's convenient to specify the launch box parameters outside of the 
  // library function and right before its invocation, to allow convenient
  // tuning.
  typedef launch_params_t<128, 8> launch_t;

  // Run the simple merge. Sort values in ascending order with less_t.
  simple_merge<launch_t>(a_device.data(), a_count, b_device.data(), b_count,
    c_device.data(), less_t<int>(), context);

  // Copy the merged data back to the host and confirm that they are sorted.
  std::vector<int> c_host = from_mem(c_device);
  for(int i = 1; i < (int)c_host.size(); ++i) {
    if(c_host[i - 1] > c_host[i]) {
      printf("ERROR: Element %d is greater than element %d\n", i, i + 1);
      exit(0);
    }
  }
  printf("SUCCESS\n");

  for(int i = 0; i < min(count, 15); ++i)
    printf("%3d: %3d\n", i, c_host[i]);
  printf("...\n");

  return 0;
}
