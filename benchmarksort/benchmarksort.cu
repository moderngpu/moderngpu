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
#include <algorithm>

using namespace mgpu;

template<typename T>
void BenchmarkSortKeys(int count, int numIt, CudaContext& context) {
#ifdef _DEBUG
	numIt = 1;
#endif

	MGPU_MEM(T) source = context.GenRandom<T>(count, 0, (T)count);
	MGPU_MEM(T) data = context.Malloc<T>(count);
	std::vector<T> sourceHost;
	source->ToHost(sourceHost);

	double mgpuElapsed = 0;
	for(int it = 0; it < numIt; ++it) {
		source->ToDevice(data->get(), count);
		context.Start();
		MergesortKeys(data->get(), count, mgpu::less<T>(), context);
		mgpuElapsed += context.Split();
	}

	std::stable_sort(sourceHost.begin(), sourceHost.end());
	double cpuElapsed = context.Split();

	double bytes = sizeof(T) * count;
	double mgpuThroughput = (double)count * numIt / mgpuElapsed;
	double mgpuBandwidth = bytes * numIt / mgpuElapsed;
	double cpuThroughput = count / cpuElapsed;
	double cpuBandwidth = bytes / cpuElapsed;

	printf("%s: %9.3lf M/s  %7.3lf GB/s   %9.3lf M/s  %7.3lf GB/s\n",
		FormatInteger(count).c_str(), mgpuThroughput / 1.0e6,
		mgpuBandwidth / 1.0e9, cpuThroughput / 1.0e6,
		cpuBandwidth / 1.0e9);
	
	// Verify
	std::vector<T> host;
	data->ToHost(host);
	for(int i = 0; i < count; ++i)
		if(sourceHost[i] != host[i]) {
			printf("MISMATCH AT %d\n", i);
			exit(0);
		}
}

template<typename T>
void BenchmarkSortPairs(int count, int numIt, CudaContext& context) {
#ifdef _DEBUG
	numIt = 1;
#endif

	MGPU_MEM(T) dataKeys = context.GenRandom<T>(count, 0, (T)count);
	MGPU_MEM(T) dataVals = context.FillAscending<T>(count, 0, 1);
	std::vector<T> sourceKeysHost, sourceValsHost;
	dataKeys->ToHost(sourceKeysHost);
	dataVals->ToHost(sourceValsHost);

	double mgpuElapsed = 0;
	for(int it = 0; it < numIt; ++it) {
		dataKeys->FromHost(sourceKeysHost);
		dataVals->FromHost(sourceValsHost);

		context.Start();
		MergesortPairs(dataKeys->get(), dataVals->get(), count, mgpu::less<T>(),
			context);
		mgpuElapsed += context.Split();
	}
	
	double bytes = 2 * sizeof(T) * count;
	double mgpuThroughput = (double)count * numIt / mgpuElapsed;
	double mgpuBandwidth = bytes * numIt / mgpuElapsed;

	printf("%s: %9.3lf M/s  %7.3lf GB/s\n", FormatInteger(count).c_str(),
		mgpuThroughput / 1.0e6, mgpuBandwidth / 1.0e9);

	// Verify
	std::vector<T> hostKeys, hostVals;
	dataKeys->ToHost(hostKeys);
	dataVals->ToHost(hostVals);
	for(int i = 0; i < count; ++i) {
		if(sourceKeysHost[hostVals[i]] != hostKeys[i]) {
			printf("MISMATCH AT %d\n", i);
			exit(0);
		}
	}
}

const int Tests[][2] = { 
	{ 10000, 1000 },
	{ 50000, 1000 },
	{ 100000, 1000 },
	{ 200000, 500 },
	{ 500000, 200 },
	{ 1000000, 200 },
	{ 2000000, 200 },
	{ 5000000, 200 },
	{ 10000000, 100 },
	{ 20000000, 100 }
};
const int NumTests = sizeof(Tests) / sizeof(*Tests);

int main(int argc, char** argv) {
	ContextPtr context = CreateCudaDevice(argc, argv, true);
	
	typedef int T1;
	typedef int64 T2;
	
	printf("Benchmarking mergesort-keys on type %s.\n", TypeIdName<T1>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkSortKeys<T1>(Tests[test][0], Tests[test][1], *context);
	
	printf("\nBenchmarking mergesort-keys on type %s.\n", TypeIdName<T2>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkSortKeys<T2>(Tests[test][0], Tests[test][1], *context);
	
	printf("\nBenchmarking mergesort-pairs on type %s.\n", TypeIdName<T1>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkSortPairs<T1>(Tests[test][0], Tests[test][1], *context);
	
	printf("\nBenchmarking mergesort-pairs on type %s.\n", TypeIdName<T2>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkSortPairs<T2>(Tests[test][0], Tests[test][1], *context);
	
	return 0;
}
