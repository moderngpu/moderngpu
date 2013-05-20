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

#include "kernels/bulkremove.cuh"
#include "kernels/bulkinsert.cuh"

using namespace mgpu;

void RandomPermutation(std::vector<int>& data, int size, int count) {
	data.resize(count);
	for(int i = 0; i < count; ++i)
		data[i] = i;

	// Randomly choose one of the un-selected integers and swap it to the front.
	for(int i = 0; i < count; ++i) {
		int index = Rand(0, count - i - 1);
		std::swap(data[i], data[i + index]);
	}

	data.resize(size);
}

template<typename T>
void BenchmarkBulkRemove(int count, int numIt, CudaContext& context) {
	// Randomly remove half the inputs.
	int removeCount = count / 2;
	int keepCount = count - removeCount;
	std::vector<int> perm;
	RandomPermutation(perm, removeCount, count);
	std::sort(perm.begin(), perm.end());

	MGPU_MEM(T) data = context.FillAscending<T>(count, 0, 1);
	MGPU_MEM(int) indices = context.Malloc(perm);
	MGPU_MEM(T) dest = context.Malloc<T>(keepCount);

	std::vector<T> hostData, host;
	std::vector<int> indicesHost;
	data->ToHost(hostData);
	indices->ToHost(indicesHost);

	context.Start();
	for(int it = 0; it < numIt; ++it)
		BulkRemove(data->get(), count, indices->get(), removeCount,
			dest->get(), context);
	double elapsed = context.Split();

	double bytes = sizeof(T) * count + keepCount * sizeof(T) +
		removeCount * sizeof(int);
	double throughput = (double)count * numIt / elapsed;
	double bandwidth = bytes * numIt / elapsed;

	printf("%s: %9.3lf M/s    %7.3lf GB/s\n", FormatInteger(count).c_str(),
		throughput / 1.0e6, bandwidth / 1.0e9);
	
	// Verify
	dest->ToHost(host);
	std::vector<T> host2(keepCount);
	int index = 0, output = 0;
	for(int input = 0; input < count; ++input) {
		bool p;
		if(index >= removeCount) p = true;
		else p = input < indicesHost[index];
		
		if(p) host2[output++] = hostData[input];
		else ++index;
	}

	for(int i = 0; i < keepCount; ++i)
		if(host[i] != host2[i]) {
			printf("MISMATCH ERROR AT %d\n", i);
			exit(0);
		}
}


template<typename T>
void BenchmarkBulkInsert(int count, int numIt, CudaContext& context) {;
	int aCount = count / 2;
	int bCount = count - aCount;
	MGPU_MEM(T) aDevice = context.GenRandom<T>(aCount, 0, count);
	MGPU_MEM(int) indicesDevice = context.SortRandom<int>(aCount, 0, bCount);
	MGPU_MEM(T) bDevice = context.GenRandom<T>(bCount, 0, count);

	MGPU_MEM(T) destDevice = context.Malloc<T>(count);

	std::vector<T> aHost, bHost;
	std::vector<int> indicesHost;
	aDevice->ToHost(aHost);
	indicesDevice->ToHost(indicesHost);
	bDevice->ToHost(bHost);

	context.Start();
	for(int it = 0; it < numIt; ++it)
		BulkInsert(aDevice->get(), indicesDevice->get(), aCount, bDevice->get(),
			bCount, destDevice->get(), context);
	double elapsed = context.Split();

	int bytes = (sizeof(int) + 2 * sizeof(T)) * aCount + 2 * sizeof(T) * bCount;
	double throughput = (double)count * numIt / elapsed;
	double bandwidth = (double)bytes * numIt / elapsed;

	printf("%s: %9.3lf M/s    %7.3lf GB/s\n", FormatInteger(count).c_str(), 
		throughput / 1.0e6, bandwidth / 1.0e9);

	std::vector<T> host, host2(count);
	destDevice->ToHost(host);
	int a = 0, b = 0;
	int output = 0;
	while(output < count) {
		bool p;
		if(a >= aCount) p = false;
		else if(b >= bCount) p = true;
		else p = indicesHost[a] <= b;

		if(p) host2[output++] = aHost[a++];
		else host2[output++] = bHost[b++];
	}

	for(int i = 0; i < count; ++i)
		if(host[i] != host2[i]) {
			printf("BulkInsert error at %d\n", i);
			exit(0);
		}
}

template<typename T>
void BenchmarkCopy(int count, int numIt, CudaContext& context) {
	MGPU_MEM(T) a = context.FillAscending<T>(count, 0, 1);
	MGPU_MEM(T) b = context.Malloc<T>(count);

	context.Start();
	for(int it = 0; it < numIt; ++it) {
		cudaMemcpy(b->get(), a->get(), sizeof(T) * count, 
			cudaMemcpyDeviceToDevice);
		a.swap(b);
	}
	double elapsed = context.Split();

	double bytes = sizeof(T) * 2 * count;
	double bandwidth = bytes * numIt / elapsed;
	printf("%s: %7.3lf GB/s\n", FormatInteger(count).c_str(), bandwidth / 1.0e9);

}

const int Tests[][2] = { 
	{ 10000, 2000 },
	{ 50000, 2000 },
	{ 100000, 2000 },
	{ 200000, 1000 },
	{ 500000, 500 },
	{ 1000000, 400 },
	{ 2000000, 400 },
	{ 5000000, 400 },
	{ 10000000, 300 },
	{ 20000000, 300 }
};
const int NumTests = sizeof(Tests) / sizeof(*Tests);


int main(int argc, char** argv) {
	ContextPtr context = CreateCudaDevice(argc, argv, true);

	typedef int T1;
	typedef int64 T2;

	printf("Benchmarking BulkRemove on type %s.\n", TypeIdName<T1>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkBulkRemove<T1>(Tests[test][0], Tests[test][1], *context);

	printf("Benchmarking BulkRemove on type %s.\n", TypeIdName<T2>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkBulkRemove<T2>(Tests[test][0], Tests[test][1], *context);
	
	printf("Benchmarking BulkInsert on type %s.\n", TypeIdName<T1>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkBulkInsert<T1>(Tests[test][0], Tests[test][1], *context);

	printf("Benchmarking BulkInsert on type %s.\n", TypeIdName<T2>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkBulkInsert<T2>(Tests[test][0], Tests[test][1], *context);
		
	return 0;
}
