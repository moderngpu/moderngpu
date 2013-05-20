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

#include "kernels/bulkremove.cuh"

using namespace mgpu;

void RandomPermutation(std::vector<int>& data, int size, int count) {
	data.resize(count);
	for(int i = 0; i < count; ++i)
		data[i] = i;

	for(int i = 0; i < count; ++i)
		std::swap(data[Rand(0, count - 1)], data[Rand(0, count - 1)]);

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

	printf("%s: %9.3lf M/s  %7.3lf GB/s\n", FormatInteger(count).c_str(),
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
	printf("Benchmarking BulkRemove on type %s.\n", typeid(T1).name());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkBulkRemove<T1>(Tests[test][0], Tests[test][1], *context);

	typedef int64 T2;
	printf("Benchmarking BulkRemove on type %s.\n", typeid(T2).name());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkBulkRemove<T2>(Tests[test][0], Tests[test][1], *context);

	return 0;
}
