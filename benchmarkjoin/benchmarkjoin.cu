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

#include "kernels/join.cuh"

using namespace mgpu;

template<typename T, typename Comp>
int CPUJoin(const T* a, int aCount, const T* b, int bCount, 
	std::vector<int>& aIndices, std::vector<int>& bIndices, MgpuJoinKind kind,
	Comp comp) {

	bool SupportLeft = MgpuJoinKindLeft == kind || 
		MgpuJoinKindOuter == kind;
	bool SupportRight = MgpuJoinKindRight == kind ||
		MgpuJoinKindOuter == kind;

	std::vector<int> aLB(aCount), aCounts(aCount);
	std::vector<bool> bMatches(bCount);

	// Find the lower-bound of A into B and the match of B into A.
	// This is MGPU SortedSearch.
	int aIndex = 0, bIndex = 0; 
	int bMatchCount = 0;
	while(aIndex < aCount || bIndex < bCount) {
		bool p;
		if(bIndex >= bCount) p = true;
		else if(aIndex >= aCount) p = false;
		else p = !comp(b[bIndex], a[aIndex]);

		if(p) {
			bool match = bIndex < bCount && !comp(a[aIndex], b[bIndex]);
			aLB[aIndex++] = bIndex + ((int)match<< 31);
		} else {
			bool match = aIndex && !comp(a[aIndex - 1], b[bIndex]);;
			bMatchCount += match;
			bMatches[bIndex] = match;
			++bIndex;
		}
	}
	int rightJoinTotal = SupportRight ? (bCount - bMatchCount) : 0;

	// Find the upper-bound of A into B and use to compute the outer-product 
	// counts.
	aIndex = bIndex = 0;
	while(aIndex < aCount) {
		bool p;
		if(bIndex >= bCount)
			p = true;
		else 
			p = comp(a[aIndex], b[bIndex]);

		if(p) {
			int lb = 0x7fffffff & aLB[aIndex];
			int count = bIndex - lb;
			aCounts[aIndex] = SupportLeft ? std::max(1, count) : count;
			++aIndex;
		} else
			++bIndex;
	}

	// Scan the counts
	int x = 0;
	for(int i = 0; i < aCount; ++i) {
		int temp = aCounts[i];
		aCounts[i] = x;
		x += temp;
	}

	int leftJoinTotal = x;
	int joinTotal = leftJoinTotal + rightJoinTotal;

	aIndices.resize(joinTotal);
	bIndices.resize(joinTotal);

	// Left join with load-balancing.
	aIndex = bIndex = 0;
	while(aIndex < leftJoinTotal) {
		bool p;
		if(bIndex >= aCount) p = true;
		else p = aIndex < aCounts[bIndex];

		if(!p) { 
			++bIndex;
			continue;
		}
		int src = bIndex - 1;
		int offset = aIndex - aCounts[src];

		aIndices[aIndex] = src;

		int lb = aLB[src];
		bIndices[aIndex] = (0x80000000 & lb) ? (0x7fffffff & lb) + offset : -1;

		++aIndex;
	}

	// Right join using compaction.
	if(SupportRight) {
		x = 0;
		memset(&aIndices[0] + leftJoinTotal, -1, sizeof(int) * rightJoinTotal);
		for(int i = 0; i < bCount; ++i)
			if(!bMatches[i])
				bIndices[leftJoinTotal + x++] = i;
	}

	return joinTotal;
}

template<MgpuJoinKind Kind, typename T>
void BenchmarkJoin(int count, int numIt, CudaContext& context) {
#ifdef _DEBUG
	numIt = 1;
#endif
	
	int bCount = count / 2;
	int aCount = count - bCount;
	

	MGPU_MEM(T) aKeysDevice = context.SortRandom<T>(aCount, 0, count);
	MGPU_MEM(T) bKeysDevice = context.SortRandom<T>(bCount, 0, count);

	MGPU_MEM(int) aIndicesDevice, bIndicesDevice;
	int total = 0;
	context.Start();
	for(int it = 0; it < numIt; ++it) {
		total = RelationalJoin<Kind>(aKeysDevice->get(), aCount,
			bKeysDevice->get(), bCount, &aIndicesDevice, &bIndicesDevice,
			mgpu::less<T>(), context);
	}
	double elapsed = context.Split();

	double bytes = sizeof(T) * count + 2 * sizeof(int) * total;
	double bandwidth = bytes * numIt / elapsed;

	printf("%s - %8d    %7.3lf GB/s\n", FormatInteger(count).c_str(), total,
		bandwidth / 1.0e9);

	std::vector<int> aIndices, bIndices;
	aIndicesDevice->ToHost(aIndices);
	bIndicesDevice->ToHost(bIndices);

	std::vector<T> aKeysHost, bKeysHost;
	aKeysDevice->ToHost(aKeysHost);
	bKeysDevice->ToHost(bKeysHost);

	std::vector<int> aIndices2, bIndices2;
	int total2 = CPUJoin(&aKeysHost[0], aCount, &bKeysHost[0], bCount, 
		aIndices2, bIndices2, Kind, mgpu::less<T>());

	if(total != total2) {
		printf("ERROR SIZE MISMATCH: %d expected %d\n", total, total2);
		exit(0);
	}
	for(int i = 0; i < total; ++i) {
		if(aIndices[i] != aIndices2[i]) {
			printf("A index mismatch at %d\n", i);
			exit(0);
		}
		if(bIndices[i] != bIndices2[i]) {
			printf("B index mismatch at %d\n", i);
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
	
	printf("Inner join on type %s\n", TypeIdName<T1>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkJoin<MgpuJoinKindInner, T1>(Tests[test][0], Tests[test][1],
			*context);
	
	printf("Left join on type %s\n", TypeIdName<T1>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkJoin<MgpuJoinKindLeft, T1>(Tests[test][0], Tests[test][1],
			*context);

	printf("Right join on type %s\n", TypeIdName<T1>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkJoin<MgpuJoinKindRight, T1>(Tests[test][0], Tests[test][1],
			*context);

	printf("Outer join on type %s\n", TypeIdName<T1>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkJoin<MgpuJoinKindOuter, T1>(Tests[test][0], Tests[test][1],
			*context);

	printf("Inner join on type %s\n", TypeIdName<T2>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkJoin<MgpuJoinKindInner, T2>(Tests[test][0], Tests[test][1],
			*context);
	
	printf("Left join on type %s\n", TypeIdName<T2>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkJoin<MgpuJoinKindLeft, T2>(Tests[test][0], Tests[test][1],
			*context);

	printf("Right join on type %s\n", TypeIdName<T2>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkJoin<MgpuJoinKindRight, T2>(Tests[test][0], Tests[test][1],
			*context);

	printf("Outer join on type %s\n", TypeIdName<T2>());
	for(int test = 0; test < NumTests; ++test)
		BenchmarkJoin<MgpuJoinKindOuter, T2>(Tests[test][0], Tests[test][1],
			*context);
	
	return 0;
}