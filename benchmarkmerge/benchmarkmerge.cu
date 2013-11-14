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

#include "kernels/merge.cuh"
#include <algorithm>

using namespace mgpu;

template<typename T>
void BenchmarkMergeKeys(int count, int numIt, CudaContext& context) {

#ifdef _DEBUG
	numIt = 1;
#endif
	
	int aCount = count / 2;
	int bCount = count - aCount;

	MGPU_MEM(T) a = context.SortRandom<T>(aCount, 0, (T)count);
	MGPU_MEM(T) b = context.SortRandom<T>(bCount, 0, (T)count);
	MGPU_MEM(T) c = context.Malloc<T>(count);
	std::vector<T> aHost, bHost;
	a->ToHost(aHost);
	b->ToHost(bHost);
	std::vector<T> cHost(count);
		
	// Benchmark MGPU
	context.Start();
	for(int it = 0; it < numIt; ++it)
		MergeKeys(a->get(), aCount, b->get(), bCount, c->get(), context);
	double mgpuElapsed = context.Split();
	
	// Benchmark STL
	std::merge(aHost.begin(), aHost.end(), bHost.begin(), bHost.end(), 
		cHost.begin());
	double cpuElapsed = context.Split();

	// Compare MGPU to STL.
	std::vector<T> cHost2;
	c->ToHost(cHost2);
	for(int i = 0; i < count; ++i)
		if(cHost[i] != cHost2[i]) {
			printf("MERGE ERROR AT COUNT = %d ITEM = %d\n", count, i);
			exit(0);
		}
		
	double bytes = 2 * sizeof(T) * count;
	double mgpuThroughput = count * numIt / mgpuElapsed;
	double mgpuBandwidth = bytes * numIt / mgpuElapsed;

	double cpuThroughput = count / cpuElapsed;
	double cpuBandwidth = bytes / cpuElapsed;
	
	printf("%s: %9.3lf M/s  %7.3lf GB/s   %9.3lf M/s  %7.3lf GB/s\n",
		FormatInteger(count).c_str(), 
		mgpuThroughput / 1.0e6, mgpuBandwidth / 1.0e9, 
		cpuThroughput / 1.0e6, cpuBandwidth / 1.0e9);
}

template<typename KeyType, typename ValType>
void BenchmarkMergePairs(int count, int numIt, CudaContext& context) {

#ifdef _DEBUG
	numIt = 1;
#endif

	int aCount = count / 2;
	int bCount = count - aCount;
	MGPU_MEM(KeyType) aKeys = context.SortRandom<KeyType>(aCount, 0, count);
	MGPU_MEM(KeyType) bKeys = context.SortRandom<KeyType>(bCount, 0, count);
	MGPU_MEM(KeyType) cKeys = context.Malloc<KeyType>(count);

	MGPU_MEM(ValType) aVals = context.FillAscending<ValType>(aCount, 0, 1);
	MGPU_MEM(ValType) bVals = context.FillAscending<ValType>(bCount, aCount, 1);
	MGPU_MEM(ValType) cVals = context.Malloc<ValType>(count);

	std::vector<KeyType> aHost, bHost;
	aKeys->ToHost(aHost);
	bKeys->ToHost(bHost);

	// Benchmark MGPU.
	context.Start();
	for(int it = 0; it < numIt; ++it)
		MergePairs(aKeys->get(), aVals->get(), aCount, bKeys->get(), 
			bVals->get(), bCount, cKeys->get(), cVals->get(), 
			mgpu::less<KeyType>(), context);
	double mgpuElapsed = context.Split();

	double bytes = 2 * (sizeof(KeyType)  + sizeof(ValType)) * count;
	double mgpuThroughput = count * numIt / mgpuElapsed;
	double mgpuBandwidth = bytes * numIt / mgpuElapsed;

	printf("%s: %9.3lf M/s  %7.3lf GB/s\n",
		FormatInteger(count).c_str(), 
		mgpuThroughput / 1.0e6, mgpuBandwidth / 1.0e9);

	// Verify
	std::vector<KeyType> keysHost;
	std::vector<ValType> valsHost;
	cKeys->ToHost(keysHost);
	cVals->ToHost(valsHost);

	for(int i = 0; i < count; ++i) {
		ValType val = valsHost[i];
		KeyType x = (val < aCount) ? aHost[val] : bHost[val - aCount];
		if(x != keysHost[i]) {
			printf("MISMATCH AT ELEMENT %d\n", i);
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
	
	printf("Benchmarking merge-keys on type %s.\n", TypeIdName<T1>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkMergeKeys<T1>(Tests[test][0], Tests[test][1], *context);
	
	printf("\nBenchmarking merge-pairs on type %s.\n", TypeIdName<T1>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkMergePairs<T1, T1>(Tests[test][0], Tests[test][1], *context);

	printf("\nBenchmarking merge-keys on type %s.\n", TypeIdName<T2>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkMergeKeys<T2>(Tests[test][0], Tests[test][1], *context);

	printf("\nBenchmarking merge-pairs on type %s.\n", TypeIdName<T2>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkMergePairs<T2, T2>(Tests[test][0], Tests[test][1], *context);
		
	return 0;
}
