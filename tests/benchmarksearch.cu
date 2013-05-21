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

#include "kernels/sortedsearch.cuh"
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <algorithm>

using namespace mgpu;

template<bool UpperBound, typename T, typename Comp>
void CPUSortedSearch(const T* a, int aCount, const T* b, int bCount,
	int* c, Comp comp) {

	int aIndex = 0, bIndex = 0;
	while(aIndex < aCount) {
		if(bIndex < bCount) {
			bool p = UpperBound ? comp(a[aIndex], b[bIndex]) :
				!comp(b[bIndex], a[aIndex]);
			if(!p) {
				++bIndex;
				continue;
			}
		}
		c[aIndex++] = bIndex;
	}
}


template<typename T>
void BenchmarkSearch(int count, double ratio, int numIt, CudaContext& context) {
	int bCount = (int)(count / (1 + ratio));
	int aCount = count - bCount;

	MGPU_MEM(T) a = context.SortRandom<T>(aCount, 0, (T)count);
	MGPU_MEM(T) b = context.SortRandom<T>(bCount, 0, (T)count);
	MGPU_MEM(int) c = context.Malloc<int>(aCount);

	thrust::device_ptr<T> pA(a->get()), pB(b->get());
	thrust::device_ptr<int> pC(c->get());

	std::vector<T> aHost, bHost;
	a->ToHost(aHost);
	b->ToHost(bHost);
	CudaTimer timer;

	// Benchmark thrust.
	timer.Start();
	for(int it = 0; it < numIt; ++it) {
		thrust::lower_bound(pB, pB + bCount, pA, pA + aCount, pC, 
			mgpu::less<T>());
	}
	double thrustElapsed = timer.Split();

	// Benchmark MGPU.
	for(int it = 0; it < numIt; ++it) {
		SortedSearch<false, true, false, 0>(a->get(), aCount, b->get(), bCount,
			c->get(), (int*)0, mgpu::less<T>(), context);
	}
	double mgpuElapsed = timer.Split();

	// Benchmark CPU
	std::vector<int> cHost(aCount);
	timer.Start();
	CPUSortedSearch<false>(&aHost[0], aCount, &bHost[0], bCount, &cHost[0],
		mgpu::less<T>());
	double cpuElapsed = timer.Split();

	// Verify the MGPU search.
	std::vector<int> cHost2;
	c->ToHost(cHost2);
	for(int i = 0; i < aCount; ++i) {
		if(cHost[i] != cHost2[i]) {
			printf("SEARCH MISMATCH AT %d\n", i);
			exit(0);
		}
	}
	
	double mgpuThroughput = count * numIt / mgpuElapsed / 1.0e6;
	double thrustThroughput = count * numIt / thrustElapsed / 1.0e6;
	double cpuThroughput = count / cpuElapsed / 1.0e6;
	printf("%9d: %8.2lf M/s   %8.2lf M/s   %8.2lf M/s\n", count,
		mgpuThroughput, thrustThroughput, cpuThroughput);
}

const int Tests[][2] = { 
	{ 10000, 100 },
	{ 50000, 100 },
	{ 100000, 100 },
	{ 200000, 50 },
	{ 500000, 20 },
	{ 1000000, 20 }, 
	{ 2000000, 20 },
	{ 5000000, 20 },
	{ 10000000, 10 },
	{ 20000000, 10 }
};
const int NumTests = sizeof(Tests) / sizeof(*Tests);

int main(int argc, char** argv) {
	ContextPtr context;
	CreateCudaContext(argc, argv, &context, true);

	for(int test = 0; test < NumTests; ++test)
		BenchmarkSearch<float>(Tests[test][0], 1.0, Tests[test][1], *context);
	cudaDeviceReset();

	return 0;
}
