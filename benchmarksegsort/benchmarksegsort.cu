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

#include "kernels/segmentedsort.cuh"

using namespace mgpu;

template<typename T>
void CpuSegSort(T* data, int count, const int* segs, int numSegs) {
	int i = 0, seg = 0;
	while(i < count) {
		// Find the next head flag.
		int end = (seg < numSegs) ? segs[seg++] : count;
		std::sort(data + i, data + end);
		i = end;
	}
}

template<typename T>
void BenchmarkSegsortKeys(int count, int segLen, int numIt, 
	CudaContext& context) {

	int numSegs = std::max(1, count / segLen - 1);
	MGPU_MEM(T) data = context.GenRandom<T>(count, 0, count - 1);
	MGPU_MEM(int) segments = context.SortRandom<int>(numSegs, 0, count - 1);

	std::vector<T> dataHost;
	data->ToHost(dataHost);
	
	std::vector<int> segsHost;
	segments->ToHost(segsHost);
	
	double elapsed = 0;
	for(int it = 0; it < numIt; ++it) {
		data->FromHost(dataHost);
		context.Start();
		SegSortKeysFromIndices(data->get(), count, segments->get(),	numSegs, 
			context);
		elapsed += context.Split();
	}
	std::vector<T> host;
	data->ToHost(host);
	
	double bytes = 2 * sizeof(T) * count;
	double throughput = (double)count * numIt / elapsed;
	double bandwidth = bytes * numIt / elapsed;

	printf("%s - %s: %9.3lf M/s  %7.3lf GB/s\n", FormatInteger(count).c_str(),
		FormatInteger(segLen).c_str(), throughput / 1.0e6, bandwidth / 1.0e9);

	// Verify.
	if(numSegs) CpuSegSort(&dataHost[0], count, &segsHost[0], numSegs);
	else std::sort(dataHost.begin(), dataHost.end());
	for(int i = 0; i < count; ++i)
		if(dataHost[i] != host[i]) {
			printf("MISMATCH AT %d\n", i);
			exit(0);
		}
}

template<typename T>
void BenchmarkSegsortPairs(int count, int segLen, int numIt,
	CudaContext& context) {

	int numSegs = max(0, count / segLen - 1);
	MGPU_MEM(T) keys = context.GenRandom<T>(count, 0, count - 1);
	MGPU_MEM(T) values = context.FillAscending<T>(count, 0, 1);
	MGPU_MEM(int) segments = context.SortRandom<int>(numSegs, 0, count - 1);

	std::vector<T> keysHost, valsHost;
	std::vector<int> segsHost;
	keys->ToHost(keysHost);
	values->ToHost(valsHost);
	segments->ToHost(segsHost);
	
	double elapsed = 0;
	for(int it = 0; it < numIt; ++it) {
		keys->FromHost(keysHost);
		values->FromHost(valsHost);
		
		context.Start();
		SegSortPairsFromIndices(keys->get(), values->get(), count,
			segments->get(), numSegs, context);
		elapsed += context.Split();
	}
	std::vector<T> host2;
	keys->ToHost(host2);
	
	valsHost.clear();
	values->ToHost(valsHost);   
	
	double bytes = 4 * sizeof(T) * count;
	double throughput = (double)count * numIt / elapsed;
	double bandwidth = bytes * numIt / elapsed;
	printf("%s - %s: %9.3lf M/s  %7.3lf GB/s\n", FormatInteger(count).c_str(),
		FormatInteger(segLen).c_str(), throughput / 1.0e6, bandwidth / 1.0e9);

	for(int i = 0; i < count; ++i) {
		T index = valsHost[i];
		if(keysHost[index] != host2[i]) {
			printf("VALUES MISMATCH AT %d\n", i);
			exit(0);
		} 
	}
	
	if(numSegs) CpuSegSort(&keysHost[0], count, &segsHost[0], numSegs);
	else std::sort(keysHost.begin(), keysHost.end());
	for(int i = 0; i < count; ++i) {
		T index = valsHost[i];
		if(keysHost[i] != host2[i]) {
			printf("KEYS MISMATCH AT %d\n", i);
			exit(0);
		}
	}
}
/*
const int Tests[][3] = { 
	{ 10000, 300000, 1000 },
	{ 50000, 300000, 1000 },
	{ 100000, 300000, 1000 },
	{ 200000, 300000, 500 },
	{ 500000, 300000, 200 },
	{ 1000000, 300000, 200 },
	{ 2000000, 300000, 200 },
	{ 5000000, 300000, 200 },
	{ 10000000, 300000, 100 },
	{ 20000000, 300000, 100 }
};*/

const int Tests[][3] = {
	{ 10000000, 100, 50 },
	{ 10000000, 300, 50 },
	{ 10000000, 1000, 50 }, 
	{ 10000000, 3000, 50 },
	{ 10000000, 10000, 50 },
	{ 10000000, 30000, 50 },
	{ 10000000, 100000, 50 },
	{ 10000000, 300000, 50 },
	{ 10000000, 1000000, 50 },
	{ 10000000, 3000000, 50 },
	{ 10000000, 10000000, 50 }
};
const int NumTests = sizeof(Tests) / sizeof(Tests[0]);

int main(int argc, char** argv) {
	ContextPtr context = CreateCudaDevice(argc, argv, true);
		
	typedef int T1;
	typedef int64 T2;

	printf("Segmented sort keys on type %s.\n", TypeIdName<T1>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkSegsortKeys<T1>(Tests[test][0], Tests[test][1], Tests[test][2],
			*context);
	
	printf("\nSegmented sort keys on type %s.\n", TypeIdName<T2>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkSegsortKeys<T2>(Tests[test][0], Tests[test][1], Tests[test][2],
			*context);
	
	printf("\nSegmented sort pairs on type %s.\n", TypeIdName<T1>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkSegsortPairs<T1>(Tests[test][0], Tests[test][1], Tests[test][2],
			*context);
	
	printf("\nSegmented sort pairs on type %s.\n", TypeIdName<T2>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkSegsortPairs<T2>(Tests[test][0], Tests[test][1], Tests[test][2],
			*context);

	return 0;
}
