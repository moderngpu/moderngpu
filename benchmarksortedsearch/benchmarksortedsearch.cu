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

#include "kernels/sortedsearch.cuh"

using namespace mgpu;

// Return lower-bound of A into B.
template<MgpuBounds Bounds, typename T, typename Comp>
void CPUSortedSearch(const T* a, int aCount, const T* b, int bCount, 
	int* indices, Comp comp) {
		
	int aIndex = 0, bIndex = 0;
	while(aIndex < aCount) {
		bool p;
		if(bIndex >= bCount)
			p = true;
		else 
			p = (MgpuBoundsUpper == Bounds)?
				comp(a[aIndex], b[bIndex]) :
				!comp(b[bIndex], a[aIndex]);

		if(p) indices[aIndex++] = bIndex;
		else ++bIndex;
	}
}

// Return lower-bound of A into B and upper-bound of B into A.
template<typename T, typename Comp>
void CPUSortedSearch2(const T* a, int aCount, const T* b, int bCount, 
	int* aIndices, int* bIndices, Comp comp) {

	int aIndex = 0, bIndex = 0;
	while(aIndex < aCount || bIndex < bCount) {
		bool p;
		if(bIndex >= bCount) p = true;
		else if(aIndex >= aCount) p = false;
		else p = !comp(b[bIndex], a[aIndex]);

		if(p) aIndices[aIndex++] = bIndex;
		else bIndices[bIndex++] = aIndex;
	}
}

// Return lower-bound of A into B and set the high bit if A has a match in B.
// Return upper-bound of B into A and set the high bit if B has a match in A.
template<typename T, typename Comp>
void CPUSortedSearch3(const T* a, int aCount, const T* b, int bCount,
	int* aIndices, int* bIndices, Comp comp) {

	int aIndex = 0, bIndex = 0;
	while(aIndex < aCount || bIndex < bCount) {
		bool p;
		if(bIndex >= bCount) p = true;
		else if(aIndex >= aCount) p = false;
		else p = !comp(b[bIndex], a[aIndex]);

		if(p) {
			// Compare the current key in A with the current key in B.
			bool match = bIndex < bCount && !comp(a[aIndex], b[bIndex]);
			aIndices[aIndex++] = bIndex + ((int)match<< 31);
		} else {
			// Compare the current key in B with the previous key in A.
			bool match = aIndex && !comp(a[aIndex - 1], b[bIndex]);
			bIndices[bIndex++] = aIndex + ((int)match<< 31);
		}
	}
}

template<typename T>
void BenchmarkSearch(int count, int numIt, double weight, 
	CudaContext& context) {

#ifdef _DEBUG
	numIt = 1;
#endif

	int aCount = (int)(weight * count);
	int bCount = count - aCount;
	MGPU_MEM(T) a = context.SortRandom<T>(aCount, 0, count - 1);
	MGPU_MEM(T) b = context.SortRandom<T>(bCount, 0, count - 1);
	MGPU_MEM(int) aIndices = context.Malloc<int>(aCount);

	// Find the lower-bound of A into B.
	context.Start();
	for(int it = 0; it < numIt; ++it) 
		SortedSearch<MgpuBoundsLower>(a->get(), aCount, b->get(), bCount,
			aIndices->get(), context);
	double elapsed = context.Split();

	double bytes = sizeof(T) * count + sizeof(int) * aCount;
	double bandwidth = bytes * numIt / elapsed;
	double throughput = (double)count * numIt / elapsed;
	printf("%s: %9.3lf M/s    %7.3lf GB/s\n", FormatInteger(count).c_str(),
		throughput / 1.0e6, bandwidth / 1.0e9);

	// Verify
	std::vector<T> aHost, bHost;
	std::vector<int> indicesHost;
	a->ToHost(aHost);
	b->ToHost(bHost);
	aIndices->ToHost(indicesHost);

	std::vector<int> indices2(aCount);
	CPUSortedSearch<MgpuBoundsLower>(&aHost[0], aCount, &bHost[0], bCount, 
		&indices2[0], mgpu::less<T>());

	for(int i = 0; i < aCount; ++i)
		if(indicesHost[i] != indices2[i]) {
			printf("Search mismatch at %d\n", i);
			exit(0);
		}
}

template<typename T>
void BenchmarkSearch2(int count, int numIt, double weight, 
	CudaContext& context) {

#ifdef _DEBUG
	numIt = 1;
#endif

	int aCount = (int)(weight * count);
	int bCount = count - aCount;
	MGPU_MEM(T) a = context.SortRandom<T>(aCount, 0, count - 1);
	MGPU_MEM(T) b = context.SortRandom<T>(bCount, 0, count - 1);
	MGPU_MEM(int) aIndices = context.Malloc<int>(aCount);
	MGPU_MEM(int) bIndices = context.Malloc<int>(bCount);

	// Find the lower-bound of A into B and upper-bound of B into A and flags
	// for all matches.
	context.Start();
	for(int it = 0; it < numIt; ++it) 
		SortedSearch<MgpuBoundsLower, MgpuSearchTypeIndexMatch, 
			MgpuSearchTypeIndexMatch>(a->get(), aCount, b->get(), bCount,
			aIndices->get(), bIndices->get(), context);
	double elapsed = context.Split();

	double bytes = sizeof(T) * count + sizeof(int) * count;
	double bandwidth = bytes * numIt / elapsed;
	double throughput = (double)count * numIt / elapsed;
	printf("%s: %9.3lf M/s    %7.3lf GB/s\n", FormatInteger(count).c_str(),
		throughput / 1.0e6, bandwidth / 1.0e9);
	
	// Verify
	std::vector<T> aHost, bHost;
	std::vector<int> aIndicesHost, bIndicesHost;
	a->ToHost(aHost);
	b->ToHost(bHost);
	aIndices->ToHost(aIndicesHost);
	bIndices->ToHost(bIndicesHost);

	std::vector<int> aIndices2(aCount), bIndices2(bCount);
	CPUSortedSearch3(&aHost[0], aCount, &bHost[0], bCount, 
		&aIndices2[0], &bIndices2[0], mgpu::less<T>());

	for(int i = 0; i < aCount; ++i)
		if(aIndicesHost[i] != aIndices2[i]) {
			printf("A lower-bound/match search mismatch at %d\n", i);
			exit(0);
		}

	for(int i = 0; i < bCount; ++i)
		if(bIndicesHost[i] != bIndices2[i]) {
			printf("B upper-bound/match search mismatch at %d\n", i);
			exit(0);
		}
}
const int Tests[][2] = { 
	{ 10000, 10000 },
	{ 50000, 10000 },
	{ 100000, 10000 },
	{ 200000, 5000 },
	{ 500000, 5000 },
	{ 1000000, 3000 },
	{ 2000000, 3000 },
	{ 5000000, 3000 },
	{ 10000000, 2000 },
	{ 20000000, 2000 }
};
const int NumTests = sizeof(Tests) / sizeof(*Tests);

int main(int argc, char** argv) {
	ContextPtr context = CreateCudaDevice(argc, argv, true);

	typedef int T1;
	typedef int64 T2;
	
	printf("Benchmarking sorted search on type %s.\n", TypeIdName<T1>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkSearch<T1>(Tests[test][0], Tests[test][1], .25, *context);

	printf("Benchmarking sorted search on type %s.\n", TypeIdName<T2>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkSearch<T2>(Tests[test][0], Tests[test][1], .25, *context);
	
	printf("Benchmarking sorted search (2) on type %s.\n", TypeIdName<T1>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkSearch2<T1>(Tests[test][0], Tests[test][1], .25, *context);
	
	printf("Benchmarking sorted search (2) on type %s.\n", TypeIdName<T2>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkSearch2<T2>(Tests[test][0], Tests[test][1], .25, *context);
	
	return 0;
}
