/******************************************************************************
 * Copyright (c) 2013, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 *
 * Code and text by Sean Baxter, NVIDIA Research
 * See http://nvlabs.github.io/moderngpu for repository and documentation.
 *
 ******************************************************************************/

#include "util/mgpucontext.h"
#include "device/launchbox.cuh"

using namespace mgpu;

// LaunchBox-specialized kernel Foo uses MGPU_LAUNCH_BOUNDS kernel to control
// register usage. The typename Tuning is obligatory, and must be chosen for 
// the MGPU_LAUNCH_BOUNDS and MGPU_LAUNCH_PARAMS macros to work.
template<typename Tuning>
MGPU_LAUNCH_BOUNDS void Foo() {
	typedef MGPU_LAUNCH_PARAMS Params;
	if(!blockIdx.x && !threadIdx.x)
		printf("Launch Foo<<<%d, %d>>> with NT=%d VT=%d OCC=%d\n", 
			gridDim.x, blockDim.x, Params::NT, Params::VT, Params::OCC);
}

// Use the built-in LaunchBoxVT type to specialize for NT, VT, and OCC.
void LaunchFoo(int count, CudaContext& context) {
	typedef LaunchBoxVT<
		128, 7, 4,			// sm_20  NT=128, VT=7,  OCC=4
		256, 9, 5,			// sm_30  NT=256, VT=9,  OCC=5
		256, 15, 3			// sm_35  NT=256, VT=15, OCC=3
	> Tuning;
	
	// GetLaunchParamaters returns (NT, VT) for the arch vesion of the provided
	// CudaContext. The product of these is the tile size.
	int2 launch = Tuning::GetLaunchParams(context);

	int NV = launch.x * launch.y;
	int numBlocks = MGPU_DIV_UP(count, NV);
	Foo<Tuning><<<numBlocks, launch.x>>>();
}


// LaunchBox-specialized kernel Bar introduces its own set of launch parameters.
template<int NT_, int VT_, int NumBlocks_, int P1_, typename T1_>
struct BarParams {
	enum { NT = NT_, VT = VT_, OCC = NumBlocks_, P1 = P1_ };
	typedef T1_ T1;
};
template<typename Tuning>
MGPU_LAUNCH_BOUNDS void Bar() {
	typedef MGPU_LAUNCH_PARAMS Params;
	if(!blockIdx.x && !threadIdx.x) {
		printf("Launch Bar<<<%d, %d>>> with NT=%d VT=%d OCC=%d\n",
			gridDim.x, blockDim.x, Params::NT, Params::VT, Params::OCC);
		printf("\t\tP1 = %d  sizeof(TT1) = %d\n", Params::P1, 
			sizeof(typename Params::T1));
	}	
}

void LaunchBar(int count, CudaContext& context) {
	typedef LaunchBox<
		BarParams<128, 7, 4, 20, short>,	// sm_20
		BarParams<256, 9, 5, 30, float>,	// sm_30
		BarParams<256, 15, 3, 35, double>	// sm_35
	> Tuning;
	int2 launch = Tuning::GetLaunchParams(context);

	int nv = launch.x * launch.y;
	int numBlocks = MGPU_DIV_UP(count, nv);
	Bar<Tuning><<<numBlocks, launch.x>>>();
}

int main(int argc, char** argv) { 
	ContextPtr context = CreateCudaDevice(argc, argv, true);

	printf("Launching Foo with 1000000 inputs:\n");
	LaunchFoo(1000000, *context);
	cudaDeviceSynchronize();

	printf("\nLaunching Bar with 1000000 inputs:\n");
	LaunchBar(1000000, *context);
	cudaDeviceSynchronize();

	return 0; 
} 