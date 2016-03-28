#include <moderngpu/kernel_load_balance.hxx>

using namespace mgpu;

template<typename launch_arg_t = empty_t, typename input_it, 
  typename segments_it, typename output_it>
void lbs_expand(input_it input, int count, segments_it segments,
  int num_segments, output_it output, context_t& context) {

  transform_lbs<launch_arg_t>([=]MGPU_DEVICE(int index, int seg, int rank) {
    output[index] = input[seg];
  }, count, segments, num_segments, context);
}

template<typename launch_arg_t = empty_t, typename input_it, 
  typename segments_it, typename gather_it, typename output_it>
void lbs_gather(input_it input, int count, segments_it segments,
  int num_segments, gather_it gather, output_it output, context_t& context) {

  transform_lbs<launch_arg_t>([=]MGPU_DEVICE(int index, int seg, int rank) {
    output[index] = input[gather[seg] + rank];
  }, count, segments, num_segments, context);
}

template<typename launch_arg_t = empty_t, typename input_it, 
  typename segments_it, typename scatter_it, typename output_it>
void lbs_scatter(input_it input, int count, segments_it segments,
  int num_segments, scatter_it scatter, output_it output, context_t& context) {

  transform_lbs<launch_arg_t>([=]MGPU_DEVICE(int index, int seg, int rank) {
    output[scatter[seg] + rank] = input[index];
  }, count, segments, num_segments, context);
}

template<typename launch_arg_t = empty_t, 
  typename input_it, typename segments_it, typename scatter_it,
  typename gather_it, typename output_it>
void lbs_move(input_it input, int count, segments_it segments,
  int num_segments, scatter_it scatter, gather_it gather, output_it output, 
  context_t& context) {

  transform_lbs<launch_arg_t>([=]MGPU_DEVICE(int index, int seg, int rank) {
    output[scatter[seg] + rank] = input[gather[seg] + rank];
  }, count, segments, num_segments, context);
}


int main(int argc, char** argv) {
  standard_context_t context;

  return 0;
}
