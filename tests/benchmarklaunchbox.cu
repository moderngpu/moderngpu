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

#include "kernels/mergesort.cuh"

using namespace mgpu;

// Copy MergesortKeys host function from kernels/mergesort.cuh and modify to
// expose the tuning structure.
template<typename Tuning, typename T, typename Comp>
MGPU_HOST void TuningMergesortKeys(T* data_global, int count, Comp comp,
	CudaContext& context) {
	
	int2 launch = Tuning::GetLaunchParams(context);
	
	const int NV = launch.x * launch.y;
	int numBlocks = MGPU_DIV_UP(count, NV);
	int numPasses = FindLog2(numBlocks, true);

	MGPU_MEM(T) destDevice = context.Malloc<T>(count);
	T* source = data_global;
	T* dest = destDevice->get();

	KernelBlocksort<Tuning, false>
		<<<numBlocks, launch.x, 0, context.Stream()>>>(source, (const int*)0,
		count, (1 & numPasses) ? dest : source, (int*)0, comp);
	if(1 & numPasses) std::swap(source, dest);

	for(int pass = 0; pass < numPasses; ++pass) {
		int coop = 2<< pass;
		MGPU_MEM(int) partitionsDevice = MergePathPartitions<MgpuBoundsLower>(
			source, count, source, 0, NV, coop, comp, context);
		
		KernelMerge<Tuning, false, true>
			<<<numBlocks, launch.x, 0, context.Stream()>>>(source, 
			(const int*)0, count, source, (const int*)0, 0, 
			partitionsDevice->get(), coop, dest, (int*)0, comp);
		std::swap(dest, source);
	}
}

template<typename Tuning, typename T>
void BenchmarkTunedMergesort(int count, int numIt, CudaContext& context) {
	MGPU_MEM(T) source = context.GenRandom<T>(count, 0, (T)count);
	MGPU_MEM(T) data = context.Malloc<T>(count);
	std::vector<T> sourceHost;
	source->ToHost(sourceHost);

	double mgpuElapsed = 0;
	for(int it = 0; it < numIt; ++it) {
		source->ToDevice(data->get(), count);
		context.Start();
		TuningMergesortKeys<Tuning>(data->get(), count, mgpu::less<T>(),
			context);
		mgpuElapsed += context.Split();
	}
	
	double bytes = 2 * sizeof(T) * count;
	double mgpuThroughput = (double)count * numIt / mgpuElapsed;
	double mgpuBandwidth = bytes * numIt / mgpuElapsed;

	int2 launch = Tuning::GetLaunchParams(context);
	printf("%3dx%2d - %s: %9.3lf M/s  %7.3lf GB/s\n",
		launch.x, launch.y, FormatInteger(count).c_str(),
		mgpuThroughput / 1.0e6, mgpuBandwidth / 1.0e9);
	
	// Verify
	std::sort(sourceHost.begin(), sourceHost.end());
	std::vector<T> host;
	data->ToHost(host);
	for(int i = 0; i < count; ++i)
		if(sourceHost[i] != host[i]) {
			printf("MISMATCH AT %d\n", i);
			exit(0);
		}
}

int main(int argc, char** argv) {
	ContextPtr context = CreateCudaDevice(argc, argv, true);

	const int N = 10000000;

	typedef LaunchBoxVT<128, 5, 0> Tuning1;
	typedef LaunchBoxVT<128, 7, 0> Tuning2;
	typedef LaunchBoxVT<128, 11, 0> Tuning3;
	typedef LaunchBoxVT<128, 15, 0> Tuning4;
	typedef LaunchBoxVT<128, 19, 0> Tuning5;
	typedef LaunchBoxVT<128, 23, 0> Tuning6;
	typedef LaunchBoxVT<128, 27, 0> Tuning7;
	typedef LaunchBoxVT<256, 5, 0> Tuning8;
	typedef LaunchBoxVT<256, 7, 0> Tuning9;
	typedef LaunchBoxVT<256, 11, 0> Tuning10;
	typedef LaunchBoxVT<256, 15, 0> Tuning11;
	typedef LaunchBoxVT<256, 19, 0> Tuning12;
	typedef LaunchBoxVT<256, 23, 0> Tuning13;
	typedef LaunchBoxVT<256, 27, 0> Tuning14;

	typedef int T1;
	typedef int64 T2;

	printf("Tuning for type %s.\n", TypeIdName<T1>());
	BenchmarkTunedMergesort<Tuning1, T1>(N, 200, *context);
	BenchmarkTunedMergesort<Tuning2, T1>(N, 200, *context);
	BenchmarkTunedMergesort<Tuning3, T1>(N, 200, *context);
	BenchmarkTunedMergesort<Tuning4, T1>(N, 200, *context);
	BenchmarkTunedMergesort<Tuning5, T1>(N, 200, *context);
	BenchmarkTunedMergesort<Tuning6, T1>(N, 200, *context);
	BenchmarkTunedMergesort<Tuning7, T1>(N, 200, *context);
	BenchmarkTunedMergesort<Tuning8, T1>(N, 200, *context);
	BenchmarkTunedMergesort<Tuning9, T1>(N, 200, *context);
	BenchmarkTunedMergesort<Tuning10, T1>(N, 200, *context);
	BenchmarkTunedMergesort<Tuning11, T1>(N, 200, *context);
	BenchmarkTunedMergesort<Tuning12, T1>(N, 200, *context);
	
	printf("\nTuning for type %s.\n", TypeIdName<T2>());
	BenchmarkTunedMergesort<Tuning1, T2>(N, 200, *context);
	BenchmarkTunedMergesort<Tuning2, T2>(N, 200, *context);
	BenchmarkTunedMergesort<Tuning3, T2>(N, 200, *context);
	BenchmarkTunedMergesort<Tuning4, T2>(N, 200, *context);
	BenchmarkTunedMergesort<Tuning5, T2>(N, 200, *context);
	BenchmarkTunedMergesort<Tuning6, T2>(N, 200, *context);
	BenchmarkTunedMergesort<Tuning7, T2>(N, 200, *context);
	BenchmarkTunedMergesort<Tuning8, T2>(N, 200, *context);
	BenchmarkTunedMergesort<Tuning9, T2>(N, 200, *context);
	BenchmarkTunedMergesort<Tuning10, T2>(N, 200, *context);
	BenchmarkTunedMergesort<Tuning11, T2>(N, 200, *context);
	BenchmarkTunedMergesort<Tuning12, T2>(N, 200, *context);
}
