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

#include "kernels/localitysort.cuh"

using namespace mgpu;

template<typename T>
void BenchmarkLocalitySortKeys(int count, int random, int numIt, 
	CudaContext& context) {

	std::vector<T> keysHost(count);
	for(int i = 0; i < count; ++i)
		keysHost[i] = i + Rand(0, random);

	MGPU_MEM(T) data = context.GenRandom<T>(count, 0, count);

	double elapsed = 0;
	for(int it = 0; it < numIt; ++it) {
		data->FromHost(keysHost);

		context.Start();
		LocalitySortKeys(data->get(), count, context);
		elapsed += context.Split();
	}

	double bytes = 2 * sizeof(T) * count;
	double throughput = (double)count * numIt / elapsed;
	double bandwidth = bytes * numIt / elapsed;

	printf("%s - %s: %9.3lf M/s  %7.3lf GB/s\n", FormatInteger(count).c_str(),
		FormatInteger(random).c_str(), throughput / 1.0e6, bandwidth / 1.0e9);

	// Verify
	std::vector<T> host2;
	data->ToHost(host2);
	std::stable_sort(keysHost.begin(), keysHost.end());
	for(int i = 0; i < count; ++i)
		if(host2[i] != keysHost[i]) {
			printf("KEYS MISMATCH AT %d\n", i);
			exit(0);
		}
}

template<typename KeyType, typename ValType>
void BenchmarkLocalitySortPairs(int count, int random, int numIt, 
	CudaContext& context) {

	std::vector<KeyType> keysHost(count);
	for(int i = 0; i < count; ++i)
		keysHost[i] = i + Rand(0, random);

	MGPU_MEM(KeyType) keys = context.Malloc<KeyType>(count);
	MGPU_MEM(ValType) vals = context.FillAscending<ValType>(count, 0, 1);

	std::vector<ValType> valsHost;
	vals->ToHost(valsHost);

	double elapsed = 0;
	for(int it = 0; it < numIt; ++it) {
		keys->FromHost(keysHost);
		vals->FromHost(valsHost);

		context.Start();
		LocalitySortPairs(keys->get(), vals->get(), count, context);
		elapsed += context.Split();
	}

	double bytes = 2 * (sizeof(KeyType) + sizeof(ValType)) * count;
	double throughput = (double)count * numIt / elapsed;
	double bandwidth = bytes * numIt / elapsed;
	
	printf("%s - %s: %9.3lf M/s  %7.3lf GB/s\n", FormatInteger(count).c_str(),
		FormatInteger(random).c_str(), throughput / 1.0e6, bandwidth / 1.0e9);

	// Verify.
	std::vector<KeyType> host2;
	keys->ToHost(host2);
	vals->ToHost(valsHost);
	for(int i = 0; i < count; ++i) {
		int index = valsHost[i];
		if(keysHost[index] != host2[i]) {
			printf("VALUE MISMATCH AT %d.\n", i);
			exit(0);
		}
	}

	std::stable_sort(keysHost.begin(), keysHost.end());
	for(int i = 0; i < count; ++i)
		if(keysHost[i] != host2[i]) {
			printf("KEYS MISMATCH AT %d.\n", i);
			exit(0);
		}
}

const int Tests[][3] = {
	{ 10000000, 100, 100 },
	{ 10000000, 300, 100 },
	{ 10000000, 1000, 100 }, 
	{ 10000000, 3000, 100 },
	{ 10000000, 10000, 100 },
	{ 10000000, 30000, 100 },
	{ 10000000, 100000, 100 },
	{ 10000000, 300000, 100 },
	{ 10000000, 1000000, 100 },
	{ 10000000, 3000000, 100 },
	{ 10000000, 10000000, 100 }
};
const int NumTests = sizeof(Tests) / sizeof(Tests[0]);

int main(int argc, char** argv) {
	ContextPtr context = CreateCudaDevice(argc, argv, true);

	typedef int T1;
	typedef int64 T2;

	printf("Benchmark locality sort-keys on type %s.\n", TypeIdName<T1>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkLocalitySortKeys<T1>(Tests[test][0], Tests[test][1],
			Tests[test][2], *context);

	printf("Benchmark locality sort-keys on type %s.\n", TypeIdName<T2>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkLocalitySortKeys<T2>(Tests[test][0], Tests[test][1],
			Tests[test][2], *context);

	printf("Benchmark locality sort-pairs on type %s.\n", TypeIdName<T1>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkLocalitySortPairs<T1, T1>(Tests[test][0], Tests[test][1],
			Tests[test][2], *context);

	printf("Benchmark locality sort-pairs on type %s.\n", TypeIdName<T2>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkLocalitySortPairs<T2, T2>(Tests[test][0], Tests[test][1],
			Tests[test][2], *context);
	return 0;
}
