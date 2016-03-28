#include <moderngpu/transform.hxx>
#include <cstdio>

using namespace mgpu;         // Namespace polution.

int main(int argc, char** argv) {
  standard_context_t context;

  // One of these will be aliased to sm_ptx when the device code is compiled.
  typedef launch_box_t<
    arch_20_cta<128, 8>,    // Big Fermi GF100/GF110  eg GTX 580
    arch_21_cta<128, 4>,    // Lil Fermi GF10x/GF11x  eg GTX 550
    arch_30_cta<256, 4>,    // Lil Kepler GK10x       eg GTX 680
    arch_35_cta<256, 8>,    // Big Kepler GK110+      eg GTX 780 Ti
    arch_37_cta<256, 16>,   // Huge Kepler GK210      eg Tesla K80
    arch_50_cta<256, 8>,    // Lil Maxwell GM10x      eg GTX 750
    arch_52_cta<256, 16>    // Big Maxwell GM20x      eg GTX 980 Ti
  > launch_t;

  // We use [] to request no lambda closure. We aren't using values from
  // the surrounding scope--we're only using types.
  auto k = [] MGPU_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum { nt = params_t::nt, vt = params_t::vt };

    if(!tid) printf("Standard launch box: nt = %d vt = %d\n", nt, vt);
  };
  cta_launch<launch_t>(k, 1, context);

  context.synchronize();

  return 0;
}
