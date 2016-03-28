#include <moderngpu/transform.hxx>
#include <cstdio>

using namespace mgpu;

enum modes_t {
  mode_basic = 1000,
  mode_enhanced = 2000, 
  mode_super = 3000
};

template<int nt_, modes_t mode_, typename type_t_> 
struct mode_param_t {
  // You must define nt, vt and occ (passed to __launch_bounds__) to use 
  // the launch box mechanism.
  enum { nt = nt_, vt = 1, vt0 = vt, occ = 0 };    // Required enums.
  enum { mode = mode_ };                           // Your custom enums.
  typedef type_t_ type_t;                          // Your custom types.
};

int main(int argc, char** argv) {
  standard_context_t context;

  // Define a launch box with a custom type. Use arch_xx to associate
  // a parameters type with a PTX version.
  typedef launch_box_t<
    arch_20<mode_param_t<64, mode_basic, short> >,    // HPC Fermi
    arch_35<mode_param_t<128, mode_enhanced, int> >,  // HPC Kepler 
    arch_52<mode_param_t<256, mode_super, int64_t> >  // HPC Maxwell
  > launch_t;

  auto k = [] MGPU_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum { nt = params_t::nt, mode = params_t::mode };
    typedef typename params_t::type_t type_t;

    if(!tid) 
      printf("Custom launch box: nt = %d mode = %d sizeof(type_t)=%d\n",
        nt, mode, sizeof(type_t));
  };
  cta_launch<launch_t>(k, 1, context);

  context.synchronize();

  return 0;
}