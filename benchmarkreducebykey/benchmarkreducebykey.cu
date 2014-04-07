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

#include "kernels/reducebykey.cuh"
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/pair.h>
#include <thrust/device_vector.h>

using namespace mgpu;

template<typename KeyType, typename ValType>
double BenchmarkMgpu(const KeyType* keys_global, const ValType* vals_global,
	int count, int numIt, bool preprocess, bool retrieveCount, bool copyKeys,
	KeyType* keysDest_global, ValType* valsDest_global, int* numSegments,
	CudaContext& context) {

	MGPU_MEM(int) countsDevice = context.Malloc<int>(1);

	std::auto_ptr<ReduceByKeyPreprocessData> preprocessData;
	if(preprocess) {
		ReduceByKeyPreprocess<ValType>(count, keys_global, 
			copyKeys ? keysDest_global : (KeyType*)0, mgpu::equal_to<KeyType>(),
			numSegments, (int*)0, &preprocessData, context);
	}

	context.Start();
	for(int it = 0; it < numIt; ++it) {
		if(preprocess) {
			ReduceByKeyApply(*preprocessData, vals_global, (ValType)0, 
				mgpu::plus<ValType>(), valsDest_global, context);
		} else {
			ReduceByKey(keys_global, vals_global, count, (ValType)0,
				mgpu::plus<ValType>(), mgpu::equal_to<ValType>(),
				copyKeys ? keysDest_global : (KeyType*)0, valsDest_global,
				retrieveCount ? numSegments : (int*)0, 
				retrieveCount ? (int*)0 : countsDevice->get(), context);
		}
	}
	double elapsed = context.Split();

	if(!retrieveCount && !preprocess)
		copyDtoH(numSegments, countsDevice->get(), 1);

	return elapsed;
}

template<typename KeyType, typename ValType>
void VerifyReduceByKey(const std::vector<KeyType>& keysHost,
	const std::vector<ValType>& valsHost, int numSegments,
	const KeyType* keys_global, const ValType* vals_global,
	int numSegments2) {

	if(numSegments != numSegments2) {
		printf("reduce-by-key wrong segment count. returned %d expecting %d\n",
			numSegments2, numSegments);
		exit(0);
	}

	std::vector<KeyType> keysHost2;
	copyDtoH(keysHost2, keys_global, numSegments);
	for(int i = 0; i < numSegments; ++i)
		if(keysHost[i] != keysHost2[i]) {
			printf("Key error at segment %d\n", i);
			exit(0);
		}

	std::vector<ValType> valsHost2;
	copyDtoH(valsHost2, vals_global, numSegments);
	for(int i = 0; i < numSegments; ++i)
		if(valsHost[i] != valsHost2[i]) {
			printf("Reduction error at segment %d\n", i);
			exit(0);
		}
}


template<typename KeyType, typename ValType>
void BenchmarkReduceByKey(int count, int randomSize, int numIt,
	CudaContext& context) {

#ifdef _DEBUG
	numIt = 1;
#endif

	std::vector<int> segCountsHost, csrHost;
	int total = 0;
	while(total < count) {
		int segSize = Rand(1, min(2 * randomSize, count - total));
		csrHost.push_back(total ? (csrHost.back() + segCountsHost.back()) : 0);
		segCountsHost.push_back(segSize);
		total += segSize;
	}
	int numSegments = (int)segCountsHost.size();
	
	std::vector<KeyType> segsHost(count);
	std::vector<KeyType> keysHost(numSegments);
	for(int i = 0; i < numSegments; ++i) {
		int begin = csrHost[i];
		int end = (i + 1 < numSegments) ? csrHost[i + 1] : count;
		std::fill(&segsHost[0] + begin, &segsHost[0] + end, i);
		keysHost[i] = i;
	}

	MGPU_MEM(KeyType) keysDevice = context.Malloc(segsHost);
	MGPU_MEM(KeyType) keysDestDevice = context.Malloc<KeyType>(numSegments);

	// Generate random ints as input.
	std::vector<ValType> dataHost(count);
	for(int i = 0; i < count; ++i)
		dataHost[i] = (ValType)Rand(0, 9);

	// Compute reference output.
	std::vector<ValType> resultsRef(numSegments);
	for(int seg = 0; seg < numSegments; ++seg) {
		int begin = csrHost[seg];
		int end = (seg + 1 < numSegments) ? csrHost[seg + 1] : count;
			
		ValType x = 0;
		for(int i = begin; i < end; ++i)
			x = x + dataHost[i];

		resultsRef[seg] = x;
	}

	MGPU_MEM(ValType) dataDevice = context.Malloc(dataHost);
	MGPU_MEM(ValType) resultsDevice = context.Malloc<ValType>(numSegments);
	std::vector<KeyType> keysDestHost;
	std::vector<ValType> resultsHost;

	////////////////////////////////////////////////////////////////////////////
	// MGPU reduce-by-key

	int numSegments2;
	double elapsed = BenchmarkMgpu(keysDevice->get(), dataDevice->get(), count,
		numIt, true, true, true, keysDestDevice->get(), resultsDevice->get(),
		&numSegments2, context);
	double preprocessedThroughput = (double)count * numIt / elapsed;
	VerifyReduceByKey(keysHost, resultsRef, numSegments, keysDestDevice->get(),
		resultsDevice->get(), numSegments);

	elapsed = BenchmarkMgpu(keysDevice->get(), dataDevice->get(), count,
		numIt, false, true, true, keysDestDevice->get(), resultsDevice->get(),
		&numSegments2, context);
	double retrieveCountThroughput = (double)count * numIt / elapsed;
	VerifyReduceByKey(keysHost, resultsRef, numSegments, keysDestDevice->get(),
		resultsDevice->get(), numSegments);

	elapsed = BenchmarkMgpu(keysDevice->get(), dataDevice->get(), count,
		numIt, false, false, true, keysDestDevice->get(), resultsDevice->get(),
		&numSegments2, context);
	double retrieveKeysThroughput = (double)count * numIt / elapsed;
	VerifyReduceByKey(keysHost, resultsRef, numSegments, keysDestDevice->get(),
		resultsDevice->get(), numSegments);
	
	elapsed = BenchmarkMgpu(keysDevice->get(), dataDevice->get(), count,
		numIt, false, false, false, keysDestDevice->get(), resultsDevice->get(),
		&numSegments2, context);
	double onlyReduceThroughput = (double)count * numIt / elapsed;
	VerifyReduceByKey(keysHost, resultsRef, numSegments, keysDestDevice->get(),
		resultsDevice->get(), numSegments);

	printf("%6s - %6s: %8.1lf M/s   %8.1lf M/s   %8.1lf M/s   %8.1lf M/s   ",
		FormatInteger(count).c_str(), FormatInteger(randomSize).c_str(),
		preprocessedThroughput / 1.0e6, retrieveCountThroughput / 1.0e6, 
		retrieveKeysThroughput / 1.0e6, onlyReduceThroughput / 1.0e6);
	
	////////////////////////////////////////////////////////////////////////////
	// Thrust reduce-by-key

	context.Start();
	for(int it = 0; it < numIt; ++it) {
		typedef thrust::device_ptr<KeyType> PK;
		typedef thrust::device_ptr<ValType> PV;
		
		PK keys_first(keysDevice->get());
		PK keys_last(keysDevice->get() + count);
		PV values_first(dataDevice->get());
		PK keys_output(keysDestDevice->get());
		PV values_output(resultsDevice->get());

		thrust::pair<PK, PV> result = thrust::reduce_by_key(keys_first, 
			keys_last, values_first, keys_output, values_output,
			mgpu::equal_to<KeyType>(), mgpu::plus<ValType>());

		numSegments2 = (int)(result.first - keys_output);
	}
	elapsed = context.Split();
	double thrustThroughput = (double)count * numIt / elapsed;

	VerifyReduceByKey(keysHost, resultsRef, numSegments, keysDestDevice->get(),
		resultsDevice->get(), numSegments);
	
	printf("%8.1lf M/s\n", thrustThroughput / 1.0e6);
}

const int Tests[][2] = { 
	{ 10000, 10000 },
	{ 50000, 10000 },
	{ 100000, 10000 },
	{ 200000, 5000 },
	{ 500000, 2000 },
	{ 1000000, 2000 },
	{ 2000000, 2000 },
	{ 5000000, 2000 },
	{ 10000000, 1000 },
	{ 20000000, 1000 }
};
const int NumTests = sizeof(Tests) / sizeof(*Tests); 

const int SegSizes[] = { 
	10,
	20,
	50,
	100,
	200,
	500,
	1000,
	2000,
	5000,
	10000,
	20000,
	50000,
	100000,
	200000,
	500000
};
const int NumSegSizes = sizeof(SegSizes) / sizeof(*SegSizes);

template<typename KeyType, typename ValType>
void BenchmarkReduceByKey1(CudaContext& context) {

	int avSegSize = 500;
	printf("Benchmarking reduce-by-key type (%s, %s). AvSegSize = %d.\n",
		TypeIdName<KeyType>(), TypeIdName<ValType>(), avSegSize);

	for(int test = 0; test < NumTests; ++test) {
		int count = Tests[test][0];

		BenchmarkReduceByKey<KeyType, ValType>(count, avSegSize, Tests[test][1], 
			context);

		context.GetAllocator()->Clear();
	}
	printf("\n");
}

template<typename KeyType, typename ValType>
void BenchmarkReduceByKey2(CudaContext& context) {
	int count = 20000000;
	printf("Benchmarking reduce-by-key type (%s, %s). Count = %d.\n",
		TypeIdName<KeyType>(), TypeIdName<ValType>(), count);

	for(int test = 0; test < NumSegSizes; ++test) {
		int avSegSize = SegSizes[test];

		BenchmarkReduceByKey<KeyType, ValType>(count, avSegSize, 500, context);

		context.GetAllocator()->Clear();
	}
	printf("\n");
}

int main(int argc, char** argv) {
	ContextPtr context = CreateCudaDevice(argc, argv, true);

	typedef int KeyType;
	
	BenchmarkReduceByKey1<KeyType, float>(*context);		
	BenchmarkReduceByKey1<KeyType, double>(*context);
	
	BenchmarkReduceByKey2<KeyType, float>(*context);
	BenchmarkReduceByKey2<KeyType, double>(*context);
		
	return 0;
}

