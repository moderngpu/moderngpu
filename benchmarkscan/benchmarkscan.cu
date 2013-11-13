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

#include "kernels/scan.cuh"
#include "kernels/reduce.cuh"

using namespace mgpu;

////////////////////////////////////////////////////////////////////////////////

template<typename Op>
void BenchmarkScan(int count, int numIt, typename Op::result_type identity,
	Op op, CudaContext& context) {

#ifdef _DEBUG
	numIt = 1;
#endif

	typedef typename Op::result_type T;
	MGPU_MEM(T) inputDevice = //context.GenRandom<T>(count, 0, 10);
		context.Fill<T>(count, 1);
	MGPU_MEM(T) resultDevice = context.Malloc<T>(count);
		
	// Benchmark MGPU
	context.Start();
	for(int it = 0; it < numIt; ++it)
		Scan<MgpuScanTypeExc>(inputDevice->get(), count, 
			identity, op, (T*)0, (T*)0, resultDevice->get(), context);
	double mgpuElapsed = context.Split();
	
	double bytes = (2 * sizeof(T) + sizeof(T)) * count;
	double mgpuThroughput = (double)count * numIt / mgpuElapsed;
	double mgpuBandwidth = bytes * numIt / mgpuElapsed;
	printf("%s: %9.3lf M/s  %7.3lf GB/s\n", FormatInteger(count).c_str(),
		mgpuThroughput / 1.0e6, mgpuBandwidth / 1.0e9);

	// Verify the results again the host calculation.
	std::vector<T> host;
	resultDevice->ToHost(host);

	std::vector<T> inputHost;
	inputDevice->ToHost(inputHost);

	T x = identity;
	for(int i = 0; i < count; ++i) {
		if(x != host[i]) {
			printf("ERROR AT %d\n", i);
			exit(0);
		}
		T value = inputHost[i];
		x = op(x, value);
	}
}

template<typename Op>
void BenchmarkMaxReduce(int count, int numIt, CudaContext& context) {

#ifdef _DEBUG
	numIt = 1;
#endif

	typedef typename Op::first_argument_type T;
	MGPU_MEM(T) data = context.GenRandom<T>(count, 0, count);
	MGPU_MEM(T) result = context.Malloc<T>(1);

	// Benchmark MGPU
	context.Start();
	for(int it = 0; it < numIt; ++it)
		Reduce(data->get(), count, (T)0, Op(), result->get(), (T*)0, context);
	double elapsed = context.Split();

	T resultHost;
	result->ToHost(&resultHost, 1);

	double bytes = count * sizeof(T);
	double mgpuThroughput = (double)count * numIt / elapsed;
	double mgpuBandwidth = bytes * numIt / elapsed;

	printf("%s: %9.3lf M/s  %7.3lf GB/s\n", 
		FormatInteger(count).c_str(), mgpuThroughput / 1.0e6,
		mgpuBandwidth / 1.0e9);
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
	
	{	
		typedef mgpu::plus<T1> Op1;
		typedef mgpu::plus<T2> Op2;

		printf("Benchmarking scan on type %s\n", TypeIdName<T1>());
		for(int test = 0; test < NumTests; ++test)
			BenchmarkScan<Op1>(Tests[test][0], Tests[test][1], (T1)0, Op1(),
				*context);
	
		printf("\nBenchmarking scan on type %s\n", TypeIdName<T2>());
		for(int test = 0; test < NumTests; ++test)
			BenchmarkScan<Op2>(Tests[test][0], Tests[test][1], (T2)0, Op2(),
				*context);
	}
	{
		typedef mgpu::maximum<T1> Op1;
		typedef mgpu::maximum<T2> Op2;

		printf("\nBenchmarking max-reduce on type %s\n", TypeIdName<T1>());
		for(int test = 0; test < NumTests; ++test)
			BenchmarkMaxReduce<Op1>(Tests[test][0], Tests[test][1], *context);
	
		printf("\nBenchmarking max-reduce on type %s\n", TypeIdName<T2>());
		for(int test = 0; test < NumTests; ++test)
			BenchmarkMaxReduce<Op2>(Tests[test][0], Tests[test][1], *context);
	}
	
	return 0;
} 
