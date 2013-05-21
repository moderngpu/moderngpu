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

#include <thrust/scan.h>
#include <thrust/device_ptr.h>

using namespace mgpu;

template<typename Op>
void BenchmarkScan(int count, int numIt, Op op, CudaContext& context,
	bool verify = true) {

	typedef typename Op::input_type input_type;
	typedef typename Op::value_type value_type;
	typedef typename Op::result_type result_type;
	MGPU_MEM(input_type) inputDevice = 
		context.GenRandom<input_type>(count, 0, 10);
	MGPU_MEM(result_type) resultDevice = context.Malloc<result_type>(count);
		
	context.Start();
	// Benchmark thrust
	for(int it = 0; it < numIt; ++it) {
		typedef thrust::device_ptr<input_type> P;
		thrust::exclusive_scan(P(inputDevice->get()),
			P(inputDevice->get()) + count, P(resultDevice->get()));
	}
	double thrustElapsed = context.Split();
	
	// Benchmark MGPU
	for(int it = 0; it < numIt; ++it)
		Scan<MgpuScanTypeExc>(inputDevice->get(), count, resultDevice->get(),
			op, 0, false, context);
	double mgpuElapsed = context.Split();

	double bytes = (2 * sizeof(input_type) + sizeof(value_type)) * count;
	double mgpuThroughput = (double)count * numIt / mgpuElapsed;
	double mgpuBandwidth = bytes * numIt / mgpuElapsed;
	double thrustThroughput = (double)count * numIt / thrustElapsed;
	double thrustBandwidth = bytes * numIt / thrustElapsed;

	printf("%s: %9.3lf M/s  %7.3lf GB/s   %9.3lf M/s  %7.3lf GB/s\n",
		FormatInteger(count).c_str(), mgpuThroughput / 1.0e6,
		mgpuBandwidth / 1.0e9, thrustThroughput / 1.0e6,
		thrustBandwidth / 1.0e9);

	// Verify the results again the host calculation.
	std::vector<result_type> host;
	resultDevice->ToHost(host);

	std::vector<input_type> inputHost;
	inputDevice->ToHost(inputHost);

	value_type x = op.Extract(op.Identity(), -1);
	for(int i = 0; i < count; ++i) {
		value_type value = op.Extract(inputHost[i], i);
		if(op.Combine(inputHost[i], x) != host[i]) {
			printf("ERROR AT %d\n", i);
			exit(0);
		}
		x = i ? op.Plus(x, value) : value;
	}
}

template<typename Op>
void BenchmarkMaxIndex(int count, int numIt, CudaContext& context) {
	typedef typename Op::input_type input_type;
	MGPU_MEM(input_type) data = context.GenRandom<input_type>(count, 0, count);

	// Benchmark MGPU
	typename Op::Pair pair;
	context.Start();
	for(int it = 0; it < numIt; ++it)
		pair = Reduce(data->get(), count, Op(), context);
	double elapsed = context.Split();

	double bytes = count * sizeof(input_type);
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
		typedef ScanOp<ScanOpTypeAdd, T1> Op1;
		typedef ScanOp<ScanOpTypeAdd, T2> Op2;

		printf("Benchmarking scan on type %s\n", TypeIdName<T1>());
		for(int test = 0; test < NumTests; ++test)
			BenchmarkScan<Op1>(Tests[test][0], Tests[test][1], Op1(), *context);
	
		printf("\nBenchmarking scan on type %s\n", TypeIdName<T2>());
		for(int test = 0; test < NumTests; ++test)
			BenchmarkScan<Op2>(Tests[test][0], Tests[test][1], Op2(), *context);
	}
	{
		typedef ScanOpMaxIndex<T1> Op1;
		typedef ScanOpMaxIndex<T2> Op2;

		printf("\nBenchmarking max-index on type %s\n", TypeIdName<T1>());
		for(int test = 0; test < NumTests; ++test)
			BenchmarkMaxIndex<Op1>(Tests[test][0], Tests[test][1], *context);
	
		printf("\nBenchmarking max-index on type %s\n", TypeIdName<T2>());
		for(int test = 0; test < NumTests; ++test)
			BenchmarkMaxIndex<Op2>(Tests[test][0], Tests[test][1], *context);
	}
	
	return 0;
} 
