#include <moderngpu/kernel_merge.hxx>       // merge
#include <moderngpu/operators.hxx>          // make_load_iterator
                                            // make_store_iterator
                                            // strided_iterator_t
using namespace mgpu;

// Use Binet's Formula to compute the n'th Fibonacci number in constant
// time.
// http://www.cut-the-knot.org/proofs/BinetFormula.shtml  
MGPU_HOST_DEVICE int binet_formula(int index) {
  const double phi = 1.61803398875;
  int fib = (index < 2) ? index : 
    (int)((pow(phi, index) - pow(phi, -index)) / sqrt(5.0) + .5);
  return fib;
}

int main(int argc, char** argv) {
  standard_context_t context;

  // We call the function from a load iterator. Now it looks like we're
  // accessing an array.
  auto a = make_load_iterator<int>([]MGPU_DEVICE(int index) {
    return binet_formula(index);
  });

  // Print the first 20 Fibonacci numbers.
  transform([=]MGPU_DEVICE(int index) {
    int fib = a[index];
    printf("Fibonacci %2d = %4d\n", index, fib);
  }, 20, context);
  context.synchronize();

  // Merge with numbers starting at 50 and incrementing by 500 each time.
  strided_iterator_t<int> b(50, 500);

  int a_count = 20;
  int b_count = 10;

  // Make a store iterator that stores the logarithm of the merged value.
  mem_t<double2> c_device(a_count + b_count, context);
  double2* c_data = c_device.data();

  auto c = make_store_iterator<int>([=]MGPU_DEVICE(int value, int index) {
    c_data[index] = make_double2(value, value ? log((double)value) : 0);
  });

  // Merge iterators a and b into iterator c.
  merge(a, a_count, b, b_count, c, less_t<int>(), context);

  // Print the results.
  std::vector<double2> c_host = from_mem(c_device);
  for(double2 result : c_host)
    printf("log(%4d) = %f\n", (int)result.x, result.y);

  return 0;
}
