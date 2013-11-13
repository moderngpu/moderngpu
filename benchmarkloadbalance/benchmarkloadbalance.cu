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
#include "kernels/loadbalance.cuh"

using namespace mgpu;

void CPULoadBalanceSearch(int aCount, const int* b, int bCount, int* indices) {
	int ai = 0, bi = 0;
	while(ai < aCount || bi < bCount) {
		bool p;
		if(bi >= bCount) p = true;
		else if(ai >= aCount) p = false;
		else p = ai < b[bi];	// aKey < bKey is upper-bound condition.

		if(p) indices[ai++] = bi - 1;	// subtract 1 from the upper-bound.
		else ++bi;
	}
}

void BenchmarkLoadBalance(int total, int numIt, double percentTerms,
	CudaContext& context) {

	int count = (int)((1.0 - percentTerms) * total);
	int numTerms = total - count;

	// Fill numTerms items with random integers that add up to count.
	div_t d = div(count, numTerms);
	std::vector<int> terms(numTerms);
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

	MGPU_MEM(int) countsDevice = context.Malloc(terms);
	MGPU_MEM(int) indexDevice = context.Malloc<int>(count);

	ScanExc(countsDevice->get(), numTerms, context);

	context.Start();
	for(int it = 0; it < numIt; ++it) 
		LoadBalanceSearch(count, countsDevice->get(), numTerms, 
			indexDevice->get(), context);
	double elapsed = context.Split();
	
	double bytes = sizeof(int) * total;
	double throughput = (double)total * numIt / elapsed;
	double bandwidth = bytes * numIt / elapsed;

	printf("%s - %s: %9.3lf M/s   %7.3lf GB/s\n", FormatInteger(total).c_str(),
		FormatInteger(numTerms).c_str(), throughput / 1.0e6, bandwidth / 1.0e9);
	
	std::vector<int> indexHost;
	indexDevice->ToHost(indexHost);

	std::vector<int> scanHost;
	countsDevice->ToHost(scanHost);

	std::vector<int> index2(count);
	CPULoadBalanceSearch(count, &scanHost[0], numTerms, &index2[0]);

	for(int i = 0; i < count; ++i)
		if(indexHost[i] != index2[i]) {
			printf("MISMATCH AT %d\n", i);
			exit(0);
		}
}

const int Tests[][2] = { 
	{ 10000, 50000 },
	{ 50000, 40000 },
	{ 100000, 20000 },
	{ 200000, 10000 },
	{ 500000, 10000 },
	{ 1000000, 5000 },
	{ 2000000, 5000 },
	{ 5000000, 4000 },
	{ 10000000, 3000 },
	{ 20000000, 3000 },
	{ 50000000, 2000 }
};
const int NumTests = sizeof(Tests) / sizeof(*Tests);


int main(int argc, char** argv) {
	ContextPtr context = CreateCudaDevice(argc, argv, true);

	for(int test = 0; test < NumTests; ++test) {
		BenchmarkLoadBalance(Tests[test][0], Tests[test][1], 0.25,
			*context);
	}
	printf("\n");
	for(int test = 0; test < 10; ++test) {
		double ratio = .05 + .10 * test;
		BenchmarkLoadBalance(10000000, 300, ratio, *context);
	}

	return 0;
}
