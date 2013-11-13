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

#include "kernels/scan.cuh"
#include "kernels/intervalmove.cuh"

using namespace mgpu;

template<typename T> 
void CPUIntervalExpand(int count, const int* scan, const T* data, int numTerms,
	T* dest) {

	for(int i = 0; i < numTerms; ++i) {
		int begin = scan[i];
		int end = (i < numTerms - 1) ? scan[i + 1] : count;
		std::fill(dest + begin, dest + end, data[i]);
	}
}

// Fill numTerms items with random integers that add up to count.
void GenRandomIntervals(int count, int numTerms, int* terms) {
	div_t d = div(count, numTerms);
	for(int i = 0; i < numTerms; ++i)
		terms[i] = d.quot + (i < d.rem);

	for(int i = 0; i < numTerms - 1; ++i) {
		int r = Rand(0, numTerms - i - 1);
		int x = std::min(terms[r], terms[i + r]);
		int r2 = Rand(-x, x);
		terms[r] -= r2;
		terms[i + r] += r2;
		std::swap(terms[r], terms[i + r]);
	}
}

template<typename T>
void BenchmarkIntervalExpand(int count, int numIt, int numTerms, 
	CudaContext& context) {

	std::vector<int> terms(numTerms);
	GenRandomIntervals(count, numTerms, &terms[0]);

	MGPU_MEM(int) countsDevice = context.Malloc(terms);
	MGPU_MEM(T) dataDevice = context.FillAscending<T>(numTerms, 0, 1);
	MGPU_MEM(T) destDevice = context.Malloc<T>(count);
	
	ScanExc(countsDevice->get(), numTerms, context);

	context.Start();
	for(int it = 0; it < numIt; ++it)
		IntervalExpand(count, countsDevice->get(), dataDevice->get(),
			numTerms, destDevice->get(), context);
	double elapsed = context.Split();
	
	double bytes = sizeof(T) * (count + numTerms) + sizeof(int) * numTerms;
	double throughput = (double)count * numIt / elapsed;
	double bandwidth = bytes * numIt / elapsed;

	printf("%s - %s: %9.3lf M/s    %7.3lf GB/s\n", FormatInteger(count).c_str(),
		FormatInteger(numTerms).c_str(), throughput / 1.0e6, bandwidth / 1.0e9);

	// Verify
	std::vector<T> destHost, dataHost;
	std::vector<int> scanHost;
	destDevice->ToHost(destHost);
	countsDevice->ToHost(scanHost);
	dataDevice->ToHost(dataHost);

	std::vector<T> dest2(count);
	CPUIntervalExpand(count, &scanHost[0], &dataHost[0], numTerms, 
		&dest2[0]);

	for(int i = 0; i < count; ++i)
		if(destHost[i] != dest2[i]) {
			printf("INTERVAL EXPAND MISMATCH AT %d\n", i);
			exit(0);
		}
}

template<typename T>
void BenchmarkIntervalMove(int count, int numIt, int numTerms,
	CudaContext& context) {
		
	// Create permutations for gather and scatter order.
	std::vector<int> permGather(numTerms), permScatter(numTerms);
	for(int i = 0; i < numTerms; ++i)
		permGather[i] = permScatter[i] = i;
	for(int i = 0; i < numTerms - 1; ++i) {
		int r1 = Rand(0, numTerms - i - 1);
		int r2 = Rand(0, numTerms - i - 1);
		std::swap(permGather[i], permGather[i + r1]);
		std::swap(permScatter[i], permScatter[i + r2]);
	}
	
	std::vector<int> terms(numTerms);
	GenRandomIntervals(count, numTerms, &terms[0]);
	
	// Randomly gather and scatter. Start by performing in-place move and
	// re-order those.
	std::vector<int> gather(numTerms), scatter(numTerms);
	for(int i = 0; i < numTerms; ++i) {
		gather[permGather[i]] = terms[i];
		scatter[permScatter[i]] = terms[i];
	}
	int x = 0, y = 0;
	for(int i = 0; i < numTerms; ++i) {
		int x2 = gather[i], y2 = scatter[i]; 
		gather[i] = x;
		scatter[i] = y;
		x += x2;
		y += y2;
	}

	std::vector<int> gatherHost(numTerms), scatterHost(numTerms);
	for(int i = 0; i < numTerms; ++i) {
		gatherHost[i] = gather[permGather[i]];
		scatterHost[i] = scatter[permScatter[i]];
	}
	

	MGPU_MEM(int) countsDevice = context.Malloc(terms);
	MGPU_MEM(int) gatherDevice = context.Malloc(gatherHost);
	MGPU_MEM(int) scatterDevice = context.Malloc(scatterHost);
	MGPU_MEM(T) sourceDevice = context.FillAscending<T>(count, 0, 1);
	MGPU_MEM(T) destDevice = context.Malloc<T>(count);
	
	ScanExc(countsDevice->get(), numTerms, context);

	context.Start();
	for(int it = 0; it < numIt; ++it)
		IntervalMove(count, gatherDevice->get(), scatterDevice->get(),
			countsDevice->get(), numTerms, sourceDevice->get(), 
			destDevice->get(), context);
	double elapsed = context.Split();

	double bytes = 3 * sizeof(int) * numTerms + 2 * sizeof(T) * count;
	double throughput = (double)count * numIt / elapsed;
	double bandwidth = bytes * numIt / elapsed;

	printf("%s - %s: %9.3lf M/s    %7.3lf GB/s\n", FormatInteger(count).c_str(),
		FormatInteger(numTerms).c_str(), throughput / 1.0e6, bandwidth / 1.0e9);

	// Verify
	std::vector<T> host;
	destDevice->ToHost(host);
	std::vector<T> host2(count);
	for(int i = 0; i < numTerms; ++i) {
		int count2 = terms[i];
		for(int j = 0; j < count2; ++j)
			host2[scatterHost[i] + j] = gatherHost[i] + j;
	}
	for(int i = 0; i < count; ++i)
		if(host[i] != host2[i]) {
			printf("INTERVAL MOVE MISMATCH AT %d\n", i);
			exit(0);
		}
}


const int Tests[][2] = { 
	{ 10000, 10000 },
	{ 50000, 1000 },
	{ 100000, 5000 },
	{ 200000, 5000 },
	{ 500000, 5000 },
	{ 1000000, 200 },
	{ 2000000, 200 },
	{ 5000000, 200 },
	{ 10000000, 1000 },
	{ 20000000, 1000 }
};
const int NumTests = sizeof(Tests) / sizeof(*Tests);

const int Terms[] = {
	5000000,
	2000000,
	1000000,
	500000,
	200000,
	100000,
	50000,
	20000,
	10000,
	5000,
	2000
};
const int NumTerms = sizeof(Terms) / sizeof(*Terms);

int main(int argc, char** argv) {
	ContextPtr context = CreateCudaDevice(argc, argv, true);

	typedef int T1;
	typedef int64 T2;
	
	// Average segment length is 25 elements.
	printf("Benchmarking interval expand on %s\n", TypeIdName<T1>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkIntervalExpand<T1>(Tests[test][0], Tests[test][1], 
			Tests[test][0] / 25, *context);

	printf("Benchmarking interval expand on %s\n", TypeIdName<T2>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkIntervalExpand<T2>(Tests[test][0], Tests[test][1], 
			Tests[test][0] / 25, *context);
	
	// Keep count at 10M inputs and change expand rate.
	printf("Benchmarking interval expand on %s\n", TypeIdName<T1>());
	for(int test = 0; test < NumTerms; ++test)
		BenchmarkIntervalExpand<T1>(10000000, 300, Terms[test], *context);

	printf("Benchmarking interval expand on %s\n", TypeIdName<T2>());
	for(int test = 0; test < NumTerms; ++test)
		BenchmarkIntervalExpand<T2>(10000000, 300, Terms[test], *context);
		
	// Average segment length is 25 elements.
	printf("Benchmarking interval move on %s\n", TypeIdName<T1>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkIntervalMove<T1>(Tests[test][0], Tests[test][1], 
			Tests[test][0] / 25, *context);
	
	printf("Benchmarking interval move on %s\n", TypeIdName<T2>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkIntervalMove<T2>(Tests[test][0], Tests[test][1], 
			Tests[test][0] / 25, *context);
	
	// Keep count at 10M inputs and change expand rate.
	printf("Benchmarking interval move on %s\n", TypeIdName<T1>());
	for(int test = 0; test < NumTerms; ++test)
		BenchmarkIntervalMove<T1>(10000000, 300, Terms[test], *context);

	printf("Benchmarking interval move on %s\n", TypeIdName<T2>());
	for(int test = 0; test < NumTerms; ++test)
		BenchmarkIntervalMove<T2>(10000000, 300, Terms[test], *context);
	
	return 0;
}

/*
void TestIntervalExpand(CudaContext& context) {
	const char* Alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
	const int Counts[26] = { 
		3, 1, 0, 0, 7, 3, 2, 14, 4, 6, 0, 2, 1,
		5, 3, 0, 5, 1, 6, 2, 0, 0, 9, 3, 2, 1		
	};
	MGPU_MEM(char) valuesDevice = context.Malloc(Alphabet, 26);
	MGPU_MEM(int) countsDevice = context.Malloc(Counts, 26);

	// Scan the fill counts to convert to output indices.
	int moveCount = Scan(countsDevice->get(), 26, context);

	// Allocate space for the output.
	MGPU_MEM(char) outputDevice = context.Malloc<char>(moveCount);

	// Perform the expand.
	IntervalExpand(moveCount, countsDevice->get(), valuesDevice->get(), 26, 
		outputDevice->get(), context);
	
	// Print the results.
	std::vector<char> outputHost(moveCount + 1);
	outputDevice->ToHost(&outputHost[0], moveCount); 
	printf("%s\n", &outputHost[0]);
}
*/
