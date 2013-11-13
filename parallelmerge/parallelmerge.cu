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

#include "moderngpu.cuh"

using namespace mgpu;

template<int NT, bool HasValues, typename InputIt1, typename InputIt2,
	typename AValsIt, typename OutputIt, typename ValsIt, typename Comp>
__global__ void ParallelMergeA(InputIt1 a_global, AValsIt aVals_global, 
	int aCount, InputIt2 b_global, int bCount, OutputIt dest_global,
	ValsIt vals_global, Comp comp) {

	typedef typename std::iterator_traits<InputIt1>::value_type T;

	int gid = threadIdx.x + NT * blockIdx.x;
	if(gid < aCount) {
		T aKey = a_global[gid];
		int lb = BinarySearch<MgpuBoundsLower>(b_global, bCount, aKey, comp);
		dest_global[gid + lb] = aKey;
		if(HasValues)
			vals_global[gid + lb] = aVals_global[gid];
	}
}

template<int NT, bool HasValues, typename InputIt1, typename InputIt2,
	typename BValsIt, typename OutputIt, typename ValsIt, typename Comp>
__global__ void ParallelMergeB(InputIt1 a_global, int aCount, InputIt2 b_global,
	BValsIt bVals_global, int bCount, OutputIt dest_global, 
	ValsIt vals_global, Comp comp) {

	typedef typename std::iterator_traits<InputIt2>::value_type T;

	int gid = threadIdx.x + NT * blockIdx.x;
	if(gid < bCount) {
		T bKey = b_global[gid];
		int ub = BinarySearch<MgpuBoundsUpper>(a_global, aCount, bKey, comp);
		dest_global[gid + ub] = bKey;
		if(HasValues)
			vals_global[gid + ub] = bVals_global[gid];
	}
}

template<typename InputIt1, typename InputIt2, typename OutputIt, typename Comp>
void ParallelMergeKeys(InputIt1 a_global, int aCount, InputIt2 b_global, 
	int bCount, OutputIt dest_global, Comp comp, CudaContext& context) {

	// NOTE: With NT = 512, limit of 33553920 in either array for Fermi arch.
	const int NT = 512;
	int aBlocks = MGPU_DIV_UP(aCount, NT);
	int bBlocks = MGPU_DIV_UP(bCount, NT);

	ParallelMergeA<NT, false><<<aBlocks, NT>>>(a_global, (const int*)0, aCount,
		b_global, bCount, dest_global, (int*)0, comp);

	ParallelMergeB<NT, false><<<bBlocks, NT>>>(a_global, aCount, b_global,
		(const int*)0, bCount, dest_global, (int*)0, comp);
}

template<typename InputIt1, typename InputIt2, typename AValsIt,
	typename BValsIt, typename OutputIt, typename ValsIt, typename Comp>
void ParallelMergePairs(InputIt1 a_global, AValsIt aVals_global, int aCount, 
	InputIt2 b_global, BValsIt bVals_global, int bCount, OutputIt dest_global,
	ValsIt vals_global, Comp comp, CudaContext& context) {

	// NOTE: With NT = 512, limit of 33553920 in either array for Fermi arch.
	const int NT = 512;
	int aBlocks = MGPU_DIV_UP(aCount, NT);
	int bBlocks = MGPU_DIV_UP(bCount, NT);

	ParallelMergeA<NT, true><<<aBlocks, NT>>>(a_global, aVals_global, aCount,
		b_global, bCount, dest_global, vals_global, comp);

	ParallelMergeB<NT, true><<<bBlocks, NT>>>(a_global, aCount, b_global,
		bVals_global, bCount, dest_global, vals_global, comp);
}


////////////////////////////////////////////////////////////////////////////////
// Benchmark

template<typename T>
void BenchmarkMergeKeysNaive(int count, int numIt, CudaContext& context) {
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
		ParallelMergeKeys(a->get(), aCount, b->get(), bCount, c->get(), 
			mgpu::less<T>(), context);
	double naiveElapsed = context.Split();
	
	// Verify
	std::merge(aHost.begin(), aHost.end(), bHost.begin(), bHost.end(), 
		cHost.begin());

	// Compare naive to STL.
	std::vector<T> cHost2;
	c->ToHost(cHost2);
	for(int i = 0; i < count; ++i)
		if(cHost[i] != cHost2[i]) {
			printf("MERGE ERROR AT COUNT = %d ITEM = %d\n", count, i);
			exit(0);
		}
		
	double bytes = 2 * sizeof(T) * count;
	double naiveThroughput = count * numIt / naiveElapsed;
	double naiveBandwidth = bytes * numIt / naiveElapsed;

	printf("%s: %9.3lf M/s  %7.3lf GB/s  \n",
		FormatInteger(count).c_str(), naiveThroughput / 1.0e6,
		naiveBandwidth / 1.0e9);
}

template<typename T>
void BenchmarkMergePairsNaive(int count, int numIt, CudaContext& context) {
	int aCount = count / 2;
	int bCount = count - aCount;

	MGPU_MEM(T) a = context.SortRandom<T>(aCount, 0, (T)count);
	MGPU_MEM(T) b = context.SortRandom<T>(bCount, 0, (T)count);
	MGPU_MEM(T) aVals = context.FillAscending<T>(aCount, 0, 1);
	MGPU_MEM(T) bVals = context.FillAscending<T>(bCount, aCount, 1);
	MGPU_MEM(T) c = context.Malloc<T>(count);
	MGPU_MEM(T) cVals = context.Malloc<T>(count);

	std::vector<T> aHost, bHost;
	a->ToHost(aHost);
	b->ToHost(bHost);
	std::vector<T> cHost(count);
		
	// Benchmark MGPU
	context.Start();
	for(int it = 0; it < numIt; ++it)
		ParallelMergePairs(a->get(), aVals->get(), aCount, b->get(), 
			bVals->get(), bCount, c->get(), cVals->get(), mgpu::less<T>(),
			context);
	double naiveElapsed = context.Split();
	
	// Verify
	std::merge(aHost.begin(), aHost.end(), bHost.begin(), bHost.end(), 
		cHost.begin());

	// Compare naive to STL.
	std::vector<T> cHost2;
	c->ToHost(cHost2);
	for(int i = 0; i < count; ++i)
		if(cHost[i] != cHost2[i]) {
			printf("MERGE ERROR AT COUNT = %d ITEM = %d\n", count, i);
			exit(0);
		}
		
	double bytes = 4 * sizeof(T) * count;
	double naiveThroughput = count * numIt / naiveElapsed;
	double naiveBandwidth = bytes * numIt / naiveElapsed;

	printf("%s: %9.3lf M/s  %7.3lf GB/s  \n",
		FormatInteger(count).c_str(), naiveThroughput / 1.0e6,
		naiveBandwidth / 1.0e9);
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
		BenchmarkMergeKeysNaive<T1>(Tests[test][0], Tests[test][1], *context);

	printf("Benchmarking merge-pairs on type %s.\n", TypeIdName<T1>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkMergePairsNaive<T1>(Tests[test][0], Tests[test][1], *context);
	
//	printf("\nBenchmarking merge-keys on type %s.\n", TypeIdName<T2>());
//	for(int test = 0; test < NumTests; ++test)
//		BenchmarkMergeKeysNaive<T2>(Tests[test][0], Tests[test][1], *context);
	
	return 0;
}

