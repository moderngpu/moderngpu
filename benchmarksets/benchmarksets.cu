/******************************************************************************
 * Copyright (c) 2013, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are     met:
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

#include "kernels/sets.cuh"
#include <algorithm>

using namespace mgpu;

template<typename T>
void BenchmarkSetsKeys(int count, int numIt, MgpuSetOp op,
	CudaContext& context) {

	int aCount = count / 2;
	int bCount = count - aCount;
	
	MGPU_MEM(T) a = context.SortRandom<T>(aCount, 0, count);
	MGPU_MEM(T) b = context.SortRandom<T>(bCount, 0, count);
		
	// Benchmark MGPU
	MGPU_MEM(T) results;
	int resultCount;
	context.Start();
	for(int it = 0; it < numIt; ++it) {
		results.reset();
		switch(op) {
			case MgpuSetOpIntersection:
				resultCount = SetOpKeys<MgpuSetOpIntersection, true>(a->get(),
					aCount, b->get(), bCount, &results, less<T>(), 
					context);
				break;
			case MgpuSetOpUnion:
				resultCount = SetOpKeys<MgpuSetOpUnion, true>(a->get(),
					aCount, b->get(), bCount, &results, less<T>(), 
					context);
				break;
			case MgpuSetOpDiff:
				resultCount = SetOpKeys<MgpuSetOpDiff, true>(a->get(),
					aCount, b->get(), bCount, &results, less<T>(), 
					context);
				break;
			case MgpuSetOpSymDiff:
				resultCount = SetOpKeys<MgpuSetOpSymDiff, true>(a->get(),
					aCount, b->get(), bCount, &results, less<T>(), 
					context);
				break;
		}
	}

	double elapsed = context.Split();
	double bytes = sizeof(T) * (count + resultCount);
	double throughput = (double)count * numIt / elapsed;
	double bandwidth = bytes * numIt / elapsed;

	printf("%s: %9.3lf M/s   %7.3lf GB/s\n", FormatInteger(count).c_str(),
		throughput / 1.0e6, bandwidth / 1.0e9);

	std::vector<T> aHost, bHost, host;
	a->ToHost(aHost);
	b->ToHost(bHost);
	results->ToHost(host);

	std::vector<T> host2(count);

	// Verify with STL.
	int stlCount;
	switch(op) {
		case MgpuSetOpIntersection:
			stlCount = std::set_intersection(aHost.begin(), aHost.end(), 
				bHost.begin(), bHost.end(), &host2[0]) - &host2[0];
			break;
		case MgpuSetOpUnion:
			stlCount = std::set_union(aHost.begin(), aHost.end(), 
				bHost.begin(), bHost.end(), &host2[0]) - &host2[0];
			break;
		case MgpuSetOpDiff:
			stlCount = std::set_difference(aHost.begin(), aHost.end(), 
				bHost.begin(), bHost.end(), &host2[0]) - &host2[0];
			break;
		case MgpuSetOpSymDiff:
			stlCount = std::set_symmetric_difference(aHost.begin(), aHost.end(), 
				bHost.begin(), bHost.end(), &host2[0]) - &host2[0];
			break;
	}
	host2.resize(stlCount);

	// Compare MGPU and STL results.
	if(stlCount != resultCount) {
		printf("SET RESULT SIZE MISMATCH: %d expected %d\n", resultCount,
			stlCount);
		exit(0);
	}
	for(int i = 0; i < resultCount; ++i)
		if(host[i] != host2[i]) {
			printf("SET OP ERROR COUNT = %d AT %d\n", count, i);
			exit(0);
		}
} 

template<typename T>
void BenchmarkSetsPairs(int count, int numIt, MgpuSetOp op,
	CudaContext& context) {

	int aCount = count / 2;
	int bCount = count - aCount;

	MGPU_MEM(T) aKeys = context.SortRandom<T>(aCount, 0, count);
	MGPU_MEM(T) bKeys = context.SortRandom<T>(bCount, 0, count);
	MGPU_MEM(T) aValues = context.FillAscending<T>(aCount, 0, 1);
	MGPU_MEM(T) bValues = context.FillAscending<T>(bCount, aCount, 1);
		
	// Benchmark MGPU
	MGPU_MEM(T) keys, values;
	int resultCount;
	context.Start();
	for(int it = 0; it < numIt; ++it) {
		keys.reset();
		values.reset();
		switch(op) {
			case MgpuSetOpIntersection:
				resultCount = SetOpPairs<MgpuSetOpIntersection, true>(
					aKeys->get(), aValues->get(), aCount, bKeys->get(), 
					bValues->get(), bCount, &keys, &values, mgpu::less<T>(),
					context);
				break;
			case MgpuSetOpUnion:
				resultCount = SetOpPairs<MgpuSetOpUnion, true>(
					aKeys->get(), aValues->get(), aCount, bKeys->get(), 
					bValues->get(), bCount, &keys, &values, mgpu::less<T>(),
					context);
				break;
			case MgpuSetOpDiff:
				resultCount = SetOpPairs<MgpuSetOpDiff, true>(
					aKeys->get(), aValues->get(), aCount, bKeys->get(), 
					bValues->get(), bCount, &keys, &values, mgpu::less<T>(),
					context);
				break;
			case MgpuSetOpSymDiff:
				resultCount = SetOpPairs<MgpuSetOpSymDiff, true>(
					aKeys->get(), aValues->get(), aCount, bKeys->get(), 
					bValues->get(), bCount, &keys, &values, mgpu::less<T>(),
					context);
				break;
		}
	}

	double elapsed = context.Split();
	double bytes = sizeof(T) * (count + resultCount) + 
		2 * sizeof(T) * resultCount;
	double throughput = (double)count * numIt / elapsed;
	double bandwidth = bytes * numIt / elapsed;

	printf("%s: %9.3lf M/s   %7.3lf GB/s\n", FormatInteger(count).c_str(),
		throughput / 1.0e6, bandwidth / 1.0e9);

	std::vector<T> aHost, bHost, host, valuesHost;
	aKeys->ToHost(aHost);
	bKeys->ToHost(bHost);
	keys->ToHost(host);
	values->ToHost(valuesHost);

	std::vector<T> host2(count);

	// Verify with STL.
	int stlCount;
	switch(op) {
		case MgpuSetOpIntersection:
			stlCount = std::set_intersection(aHost.begin(), aHost.end(), 
				bHost.begin(), bHost.end(), &host2[0]) - &host2[0];
			break;
		case MgpuSetOpUnion:
			stlCount = std::set_union(aHost.begin(), aHost.end(), 
				bHost.begin(), bHost.end(), &host2[0]) - &host2[0];
			break;
		case MgpuSetOpDiff:
			stlCount = std::set_difference(aHost.begin(), aHost.end(), 
				bHost.begin(), bHost.end(), &host2[0]) - &host2[0];
			break;
		case MgpuSetOpSymDiff:
			stlCount = std::set_symmetric_difference(aHost.begin(), aHost.end(), 
				bHost.begin(), bHost.end(), &host2[0]) - &host2[0];
			break;
	}
	host2.resize(stlCount);

	// Compare MGPU and STL results.
	if(stlCount != resultCount) {
		printf("SET RESULT SIZE MISMATCH: %d expected %d\n", resultCount,
			stlCount);
		exit(0);
	}
	for(int i = 0; i < resultCount; ++i)
		if(host[i] != host2[i]) {
			printf("SET OP ERROR COUNT = %d AT %d\n", count, i);
			exit(0);
		}
	for(int i = 0; i < resultCount; ++i) {
		int index = (int)valuesHost[i];
		T key = (index < aCount) ? aHost[index] : bHost[index - aCount];
		if(key != host[i]) {
			printf("SET-PAIRS VALUES MISMATCH: %d\n", i);
			exit(0);
		}
	}
}

const int Tests[][2] = { 
	{ 10000, 10000 },
	{ 50000, 10000 },
	{ 100000, 10000 },
	{ 200000, 5000 },
	{ 500000, 2000 },
	{ 1000000, 1000 },
	{ 2000000, 1000 },
	{ 5000000, 1000 },
	{ 10000000, 500 },
	{ 20000000, 500 }
};
const int NumTests = sizeof(Tests) / sizeof(*Tests);

int main(int argc, char** argv) {
	ContextPtr context = CreateCudaDevice(argc, argv, true);

	typedef int T1;
	typedef int64 T2;

	printf("Benchmarking sets-keys on type %s\n", TypeIdName<T1>());
	
	printf("INTERSECTION:\n");
	for(int test = 0; test < NumTests; ++test)
		BenchmarkSetsKeys<T1>(Tests[test][0], Tests[test][1],
			MgpuSetOpIntersection, *context);
	
	printf("UNION:\n");
	for(int test = 0; test < NumTests; ++test)
		BenchmarkSetsKeys<T1>(Tests[test][0], Tests[test][1],
			MgpuSetOpUnion, *context);

	printf("DIFFERENCE:\n");
	for(int test = 0; test < NumTests; ++test)
		BenchmarkSetsKeys<T1>(Tests[test][0], Tests[test][1],
			MgpuSetOpDiff, *context);

	printf("SYMMETRIC DIFFERENCE:\n");
	for(int test = 0; test < NumTests; ++test)
		BenchmarkSetsKeys<T1>(Tests[test][0], Tests[test][1],
			MgpuSetOpSymDiff, *context);
			
	/*
	printf("\nBenchmarking sets-keys on type %s\n", TypeIdName<T2>());
	printf("INTERSECTION:\n");
	for(int test = 0; test < NumTests; ++test)
		BenchmarkSetsKeys<T2>(Tests[test][0], Tests[test][1],
			MgpuSetOpIntersection, *context);
	
	printf("UNION:\n");
	for(int test = 0; test < NumTests; ++test)
		BenchmarkSetsKeys<T2>(Tests[test][0], Tests[test][1],
			MgpuSetOpUnion, *context);

	printf("DIFFERENCE:\n");
	for(int test = 0; test < NumTests; ++test)
		BenchmarkSetsKeys<T2>(Tests[test][0], Tests[test][1],
			MgpuSetOpDiff, *context);

	printf("SYMMETRIC DIFFERENCE:\n");
	for(int test = 0; test < NumTests; ++test)
		BenchmarkSetsKeys<T2>(Tests[test][0], Tests[test][1],
			MgpuSetOpSymDiff, *context);
			
	printf("\nBenchmarking sets-pairs on type %s\n", TypeIdName<T1>());
	printf("INTERSECTION:\n");
	for(int test = 0; test < NumTests; ++test)
		BenchmarkSetsPairs<T1>(Tests[test][0], Tests[test][1],
			MgpuSetOpIntersection, *context);
	
	printf("UNION:\n");
	for(int test = 0; test < NumTests; ++test)
		BenchmarkSetsPairs<T1>(Tests[test][0], Tests[test][1],
			MgpuSetOpUnion, *context);

	printf("DIFFERENCE:\n");
	for(int test = 0; test < NumTests; ++test)
		BenchmarkSetsPairs<T1>(Tests[test][0], Tests[test][1],
			MgpuSetOpDiff, *context);

	printf("SYMMETRIC DIFFERENCE:\n");
	for(int test = 0; test < NumTests; ++test)
		BenchmarkSetsPairs<T1>(Tests[test][0], Tests[test][1],
			MgpuSetOpSymDiff, *context);
	
	printf("\nBenchmarking sets-pairs on type %s\n", TypeIdName<T2>());
	printf("INTERSECTION:\n");
	for(int test = 0; test < NumTests; ++test)
		BenchmarkSetsPairs<T2>(Tests[test][0], Tests[test][1],
			MgpuSetOpIntersection, *context);

	printf("UNION:\n");
	for(int test = 0; test < NumTests; ++test)
		BenchmarkSetsPairs<T2>(Tests[test][0], Tests[test][1],
			MgpuSetOpUnion, *context);

	printf("DIFFERENCE:\n");
	for(int test = 0; test < NumTests; ++test)
		BenchmarkSetsPairs<T2>(Tests[test][0], Tests[test][1],
			MgpuSetOpDiff, *context);
			
	printf("SYMMETRIC DIFFERENCE:\n");
	for(int test = 0; test < NumTests; ++test)
		BenchmarkSetsPairs<T2>(Tests[test][0], Tests[test][1],
			MgpuSetOpSymDiff, *context);
			*/
	return 0;
}
