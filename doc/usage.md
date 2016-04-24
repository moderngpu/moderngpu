# Usage tips

* Always compile with `-Xptxas="-v"`. This is a flag to the PTX assembler asking for verbose output. 
  ```
ptxas info    : Compiling entry function '_ZN4mgpu16launch_box_cta_kINS_15launch_params_tILi16ELi1ELi1ELi0EEEZ13simple_reduceILi16EPiS4_EvT0_T1_RNS_9context_tEEUnvdl2_PFvPiPiRN4mgpu9context_tEE13simple_reduceILi16EPiPiE1_PiPiEEvT0_' for 'sm_35'
ptxas info    : Function properties for _ZN4mgpu16launch_box_cta_kINS_15launch_params_tILi16ELi1ELi1ELi0EEEZ13simple_reduceILi16EPiS4_EvT0_T1_RNS_9context_tEEUnvdl2_PFvPiPiRN4mgpu9context_tEE13simple_reduceILi16EPiPiE1_PiPiEEvT0_
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, 64 bytes smem, 336 bytes cmem[0]
```
  This is the ptx verbose output for the intra-CTA reducer in the tutorial `tut_02_cta_launch.cu`. It only uses 8 registers and 64 bytes of shared memory. Occupancy is limited by max threads per SM (2048 on recent architectures), which means the user should increase grain size `vt` to do more work per thread. 

  If the PTX assembler's output lists any byte's spilled, it is likely that the kernel is attempting to dynamically index an array that was intended to sit in register. Only random-access memories like shared, local and device memory support random access. Registers must be accessed by name, so array accesses must be made with static indices, either hard-coded or produced from the indices of compile-time unrolled loops. CUDA has its own mechanism `#pragma unroll` for unrolling loops. Unfortunately this mechanism is just a hint, and the directive can be applied to many kinds of loops that do not unroll. Use moderngpu's `iterate<>` template to guarantee loop unrolling. 
