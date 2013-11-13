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

#include "moderngpu.cuh"		// Include all MGPU kernels.

using namespace mgpu;

////////////////////////////////////////////////////////////////////////////////
// Scan

void DemoScan(CudaContext& context) {
	printf("\n\nREDUCTION AND SCAN DEMONSTRATION:\n\n");

	// Generate 100 random integers between 0 and 9.
	int N = 100;
	MGPU_MEM(int) data = context.GenRandom<int>(N, 0, 9);
	printf("Input array:\n");
	PrintArray(*data, "%4d", 10);
	
	// Run a global reduction.
	int total = Reduce(data->get(), N, context);
	printf("Reduction total: %d\n\n", total);
	
	// Run an exclusive scan.
	ScanExc(data->get(), N, &total, context);
	printf("Exclusive scan:\n");
	PrintArray(*data, "%4d", 10);
	printf("Scan total: %d\n", total);
}

////////////////////////////////////////////////////////////////////////////////
// Max-reduce

void DemoMaxReduce(CudaContext& context) {
	printf("\n\nMAX-REDUCE DEMONSTRATION:\n\n");

	// Generate 100 random integers between 0 and 999.
	int N = 100;
	MGPU_MEM(int) data = context.GenRandom<int>(N, 0, 999);
	printf("Input array:\n");
	PrintArray(*data, "%4d", 10);

	// Run a max inclusive scan.
	MGPU_MEM(int) scan = context.Malloc<int>(N);
	Scan<MgpuScanTypeInc>(data->get(), N, INT_MIN, mgpu::maximum<int>(),
		(int*)0, (int*)0, scan->get(), context);
	printf("\nInclusive max scan:\n");
	PrintArray(*scan, "%4d", 10);

	// Run a global reduction.
	int reduce;
	Reduce(data->get(), N, INT_MIN, mgpu::maximum<int>(), (int*)0, &reduce,
		context);
	printf("\nMax reduction: %d.\n", reduce);
}

////////////////////////////////////////////////////////////////////////////////
// BulkRemove

void DemoBulkRemove(CudaContext& context) {
	printf("\n\nBULK REMOVE DEMONSTRATION:\n\n");

	// Use CudaContext::FillAscending to generate 100 integers between 0 and 99.
	int N = 100;
	MGPU_MEM(int) data = context.FillAscending<int>(N, 0, 1);

	printf("Input array:\n");
	PrintArray(*data, "%4d", 10);

	// Remove every 3rd element from the exclusive scan. Use
	// CudaContext::FillAscending to generate removal indices for every 3rd
	// integer between 0 and 99.
	int RemoveCount = MGPU_DIV_UP(N, 3);
	MGPU_MEM(int) remove = context.FillAscending(RemoveCount, 0, 3);
	MGPU_MEM(int) data2 = context.Malloc<int>(N - RemoveCount);

	BulkRemove(data->get(), N, remove->get(), RemoveCount, data2->get(),
		context);
	printf("\nRemoving every 3rd element:\n");
	PrintArray(*data2, "%4d", 10);
}

////////////////////////////////////////////////////////////////////////////////
// BulkInsert

void DemoBulkInsert(CudaContext& context) {
	printf("\n\nBULK INSERT DEMONSTRATION:\n\n");

	// Use CudaContext::FillAscending to generate 100 integers between 0 and 99.
	int N = 100;
	MGPU_MEM(int) data = context.FillAscending<int>(N, 0, 1);

	printf("Input array:\n");
	PrintArray(*data, "%4d", 10);

	// Insert new elements before every 5 input starting at index 2.
	// Use step_iterator for insertion positions and content.
	int InsertCount = MGPU_DIV_UP(N - 2, 5);
	MGPU_MEM(int) data2 = context.Malloc<int>(N + InsertCount);
	mgpu::step_iterator<int> insertData(1000, 10);
	mgpu::step_iterator<int> insertIndices(2, 5);

	BulkInsert(insertData, insertIndices, InsertCount, data->get(), N, 
		data2->get(), context);

	printf("\nInserting before every 5th element starting at item 2:\n");
	PrintArray(*data2, "%4d", 10);
}

////////////////////////////////////////////////////////////////////////////////
// MergeKeys

void DemoMergeKeys(CudaContext& context) {
	printf("\n\nMERGE KEYS DEMONSTRATION:\n\n");
	
	// Use CudaContext::SortRandom to generate 100 sorted random integers 
	// between 0 and 99.
	int N = 100;
	MGPU_MEM(int) aData = context.SortRandom<int>(N, 0, 99);
	MGPU_MEM(int) bData = context.SortRandom<int>(N, 0, 99);

	printf("A:\n");
	PrintArray(*aData, "%4d", 10);
	printf("\nB:\n");
	PrintArray(*bData, "%4d", 10);

	// Merge the two sorted sequences into one.
	MGPU_MEM(int) cData = context.Malloc<int>(2 * N);
	MergeKeys(aData->get(), N, bData->get(), N, cData->get(), context);

	printf("\nMerged array:\n");
	PrintArray(*cData, "%4d", 10);
}

////////////////////////////////////////////////////////////////////////////////
// MergePairs

void DemoMergePairs(CudaContext& context) {
	printf("\n\nMERGE PAIRS DEMONSTRATION:\n\n");

	int N = 100;
	MGPU_MEM(int) aKeys = context.SortRandom<int>(N, 0, 99);
	MGPU_MEM(int) bKeys = context.SortRandom<int>(N, 0, 99);
	MGPU_MEM(int) aVals = context.FillAscending<int>(N, 0, 1);
	MGPU_MEM(int) bVals = context.FillAscending<int>(N, N, 1);

	printf("A:\n");
	PrintArray(*aKeys, "%4d", 10);
	printf("\nB:\n");
	PrintArray(*bKeys, "%4d", 10);

	// Merge the two sorted sequences into one.
	MGPU_MEM(int) cKeys = context.Malloc<int>(2 * N);
	MGPU_MEM(int) cVals = context.Malloc<int>(2 * N);
	MergePairs(aKeys->get(), aVals->get(), N, bKeys->get(), bVals->get(), N,
		cKeys->get(), cVals->get(), context);

	printf("\nMerged keys:\n");
	PrintArray(*cKeys, "%4d", 10);
	printf("\nMerged values (0-99 are A indices, 100-199 are B indices).\n");
	PrintArray(*cVals, "%4d", 10);
}

////////////////////////////////////////////////////////////////////////////////
// SortKeys

void DemoSortKeys(CudaContext& context) {
	printf("\n\nSORT KEYS DEMONSTRATION:\n\n");

	// Use CudaContext::GenRandom to generate 100 random integers between 0 and
	// 199.
	int N = 100;
	MGPU_MEM(int) data = context.GenRandom<int>(N, 0, 99);
	
	printf("Input:\n");
	PrintArray(*data, "%4d", 10);

	// Mergesort keys.
	MergesortKeys(data->get(), N, context);

	printf("\nSorted output:\n");
	PrintArray(*data, "%4d", 10);
}

////////////////////////////////////////////////////////////////////////////////
// SortPairs

void DemoSortPairs(CudaContext& context) {
	printf("\n\nSORT PAIRS DEMONSTRATION:\n\n");

	// Use CudaContext::GenRandom to generate 100 random integers between 0 and
	// 99.
	int N = 100;
	MGPU_MEM(int) keys = context.GenRandom<int>(N, 0, 99);
	MGPU_MEM(int) vals = context.FillAscending<int>(N, 0, 1);

	printf("Input keys:\n");
	PrintArray(*keys, "%4d", 10);

	// Mergesort pairs.
	MergesortPairs(keys->get(), vals->get(), N, context);

	printf("\nSorted keys:\n");
	PrintArray(*keys, "%4d", 10);

	printf("\nSorted values:\n");
	PrintArray(*vals, "%4d", 10);
}

////////////////////////////////////////////////////////////////////////////////
// SegSortKeys

void DemoSegSortKeys(CudaContext& context) {
	printf("\n\nSEG-SORT KEYS DEMONSTRATION:\n\n");

	// Use CudaContext::GenRandom to generate 100 random integers between 0 and
	// 9.
	int N = 100;
	MGPU_MEM(int) keys = context.GenRandom<int>(N, 0, 99);

	// Define 10 segment heads (for 11 segments in all).
	const int NumSegs = 10;
	const int SegHeads[NumSegs] = { 4, 19, 22, 56, 61, 78, 81, 84, 94, 97 };
	MGPU_MEM(int) segments = context.Malloc(SegHeads, 10);

	printf("Input keys:\n");
	PrintArray(*keys, "%4d", 10);

	printf("\nSegment heads:\n");
	PrintArray(*segments, "%4d", 10);

	// Sort within segments.
	SegSortKeysFromIndices(keys->get(), N, segments->get(), NumSegs, context);

	printf("\nSorted data (segment heads are marked by *):\n");
	PrintArrayOp(*keys, FormatOpMarkArray(" %c%2d", SegHeads, NumSegs), 10);
}

////////////////////////////////////////////////////////////////////////////////
// SegSortPairs

void DemoSegSortPairs(CudaContext& context) {
	printf("\n\nSEG-SORT PAIRS DEMONSTRATION:\n\n");

	// Use CudaContext::GenRandom to generate 100 random integers between 0 and
	// 9.
	int N = 100;
	MGPU_MEM(int) keys = context.GenRandom<int>(N, 0, 99);

	// Fill values with ascending integers.
	MGPU_MEM(int) values = context.FillAscending<int>(N, 0, 1);

	// Define 10 segment heads (for 11 segments in all).
	const int NumSegs = 10;
	const int SegHeads[NumSegs] = { 4, 19, 22, 56, 61, 78, 81, 84, 94, 97 };
	MGPU_MEM(int) segments = context.Malloc(SegHeads, 10);

	printf("Input keys:\n");
	PrintArray(*keys, "%4d", 10);

	printf("\nSegment heads:\n");
	PrintArray(*segments, "%4d", 10);

	// Sort within segments.
	SegSortPairsFromIndices(keys->get(), values->get(), N, segments->get(), 
		NumSegs, context);

	printf("\nSorted data (segment heads are marked by *):\n");
	PrintArrayOp(*keys, FormatOpMarkArray(" %c%2d", SegHeads, NumSegs), 10);

	printf("\nSorted indices (segment heads are marked by *):\n");
	PrintArrayOp(*values, FormatOpMarkArray(" %c%2d", SegHeads, NumSegs), 10);
}

////////////////////////////////////////////////////////////////////////////////
// LocalitySortKeys

void DemoLocalitySortKeys(CudaContext& context) {
	printf("\n\nLOCALITY SORT KEYS DEMONSTRATION:\n\n");

	// Generate keys that are roughly sorted but with added noise.
	int N = 100;
	std::vector<int> keysHost(N);
	for(int i = 0; i < N; ++i) 
		keysHost[i] = i + Rand(0, 25);

	MGPU_MEM(int) keys = context.Malloc(keysHost);

	printf("Input keys:\n");
	PrintArray(*keys, "%4d", 10);

	// Sort by exploiting locality.
	LocalitySortKeys(keys->get(), N, context);

	printf("\nSorted data:\n");
	PrintArray(*keys, "%4d", 10);;
}

////////////////////////////////////////////////////////////////////////////////
// LocalitySortPairs

void DemoLocalitySortPairs(CudaContext& context) {
	printf("\n\nLOCALITY SORT PAIRS DEMONSTRATION:\n\n");

	// Generate keys that are roughly sorted but with added noise.
	int N = 100;
	std::vector<int> keysHost(N);
	for(int i = 0; i < N; ++i) 
		keysHost[i] = i + Rand(0, 25);

	MGPU_MEM(int) keys = context.Malloc(keysHost);
	MGPU_MEM(int) values = context.FillAscending<int>(N, 0, 1);

	printf("Input keys:\n");
	PrintArray(*keys, "%4d", 10);

	// Sort by exploiting locality.
	LocalitySortPairs(keys->get(), values->get(), N, context);

	printf("\nSorted data:\n");
	PrintArray(*keys, "%4d", 10);
	
	printf("\nSorted indices:\n");
	PrintArray(*values, "%4d", 10);
}

////////////////////////////////////////////////////////////////////////////////
// SortedSearch

void DemoSortedSearch(CudaContext& context) {
	printf("\n\nSORTED SEARCH DEMONSTRATION:\n\n");

	// Use CudaContext::SortRandom to generate a haystack of 200 random integers
	// between 0 and 999 and an array of 100 needles in the same range.
	int HaystackSize = 200;
	int NeedlesSize = 100;
	MGPU_MEM(int) haystack = context.SortRandom<int>(HaystackSize, 0, 299);
	MGPU_MEM(int) needles = context.SortRandom<int>(NeedlesSize, 0, 299);

	printf("Haystack array:\n");
	PrintArray(*haystack, "%4d", 10);
	printf("\nNeedles array:\n");
	PrintArray(*needles, "%4d", 10);

	// Run a vectorized sorted search to find lower bounds.
	SortedSearch<MgpuBoundsLower>(needles->get(), NeedlesSize, haystack->get(),
		HaystackSize, needles->get(), context);

	printf("\nLower bound array:\n");
	PrintArray(*needles, "%4d", 10);
}

void DemoSortedSearch2(CudaContext& context) {
	printf("\n\nSORTED SEARCH DEMONSTRATION (2):\n\n");

	int ACount = 100;
	int BCount = 100;
	MGPU_MEM(int) aData = context.SortRandom<int>(ACount, 0, 299);
	MGPU_MEM(int) bData = context.SortRandom<int>(BCount, 0, 299);
	MGPU_MEM(int) aIndices = context.Malloc<int>(ACount);
	MGPU_MEM(int) bIndices = context.Malloc<int>(BCount);

	printf("A array:\n");
	PrintArray(*aData, "%4d", 10);
	printf("\nB array:\n");
	PrintArray(*bData, "%4d", 10);

	// Run a vectorized sorted search to find lower bounds.
	SortedSearch<MgpuBoundsLower, MgpuSearchTypeIndexMatch,
		MgpuSearchTypeIndexMatch>(aData->get(), ACount, bData->get(), BCount,
		aIndices->get(), bIndices->get(), context);

	printf("\nLower bound of A into B (* for match):\n");
	PrintArrayOp(*aIndices, FormatOpMaskBit("%c%3d"), 10);
	printf("\nUpper bound of B into A (* for match):\n");
	PrintArrayOp(*bIndices, FormatOpMaskBit("%c%3d"), 10);
}

////////////////////////////////////////////////////////////////////////////////
// LoadBalancingSearch

void DemoLBS(CudaContext& context) {
	printf("\n\nLOAD-BALANCING SEARCH DEMONSTRATION:\n\n");
	
	// Use CudaContext::GenRandom to generate work counts between 0 and 5.
	int N = 50;
	MGPU_MEM(int) counts = context.GenRandom<int>(N, 0, 5);
	
	printf("Object counts\n");
	PrintArray(*counts, "%4d", 10);

	// Scan the counts.
	int total;
	ScanExc(counts->get(), N, &total, context);
	printf("\nScan of object counts:\n");
	PrintArray(*counts, "%4d", 10);
	printf("Total: %4d\n", total);

	// Allocate space for the object references and run load-balancing search.
	MGPU_MEM(int) refsData = context.Malloc<int>(total);
	LoadBalanceSearch(total, counts->get(), N, refsData->get(), context);

	printf("\nObject references:\n");
	PrintArray(*refsData, "%4d", 10);
}

////////////////////////////////////////////////////////////////////////////////
// IntervalExpand

void DemoIntervalExpand(CudaContext& context) {
	printf("\n\nINTERVAL-EXPAND DEMONSTRATION:\n\n");

	const int NumInputs = 20;
	const int Counts[NumInputs] = { 
		2, 5, 7, 16, 0, 1, 0, 0, 14, 10, 
		3, 14, 2, 1, 11, 2, 1, 0, 5, 6 
	};
	const int Inputs[NumInputs] = {
		1, 1, 2, 3, 5, 8, 13, 21, 34, 55,
		89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765
	};
	printf("Expand counts:\n");
	PrintArray(Counts, NumInputs, "%4d", 10);

	printf("\nExpand values:\n");
	PrintArray(Inputs, NumInputs, "%4d", 10);

	MGPU_MEM(int) countsDevice = context.Malloc(Counts, NumInputs);
	int total;
	ScanExc(countsDevice->get(), NumInputs, &total, context);

	MGPU_MEM(int) fillDevice = context.Malloc(Inputs, NumInputs);

	MGPU_MEM(int) dataDevice = context.Malloc<int>(total);
	IntervalExpand(total, countsDevice->get(), fillDevice->get(), NumInputs, 
		dataDevice->get(), context);
	
	printf("\nExpanded data:\n");
	PrintArray(*dataDevice, "%4d", 10);
}

////////////////////////////////////////////////////////////////////////////////
// IntervalMove

void DemoIntervalMove(CudaContext& context) {
	printf("\n\nINTERVAL-MOVE DEMONSTRATION:\n\n");

	const int NumInputs = 20;
	const int Counts[NumInputs] = {
		3, 9, 1, 9, 8, 5, 10, 2, 5, 2,
		8, 6, 5, 2, 4, 0, 8, 2, 5, 6
	};
	const int Gather[NumInputs] = {
		75, 86, 17, 2, 67, 24, 37, 11, 95, 35,
		52, 18, 47, 0, 13, 75, 78, 60, 62, 29
	};
	const int Scatter[NumInputs] = {
		10, 80, 99, 27, 41, 71, 15, 0, 36, 13,
		89, 49, 66, 97, 76, 76, 2, 25, 61, 55
	};

	printf("Interval counts:\n");
	PrintArray(Counts, NumInputs, "%4d", 10);

	printf("\nInterval gather:\n");
	PrintArray(Gather, NumInputs, "%4d", 10);

	printf("\nInterval scatter:\n");
	PrintArray(Scatter, NumInputs, "%4d", 10);

	MGPU_MEM(int) countsDevice = context.Malloc(Counts, NumInputs);
	MGPU_MEM(int) gatherDevice = context.Malloc(Gather, NumInputs);
	MGPU_MEM(int) scatterDevice = context.Malloc(Scatter, NumInputs);
	int total;
	ScanExc(countsDevice->get(), NumInputs, &total, context);

	MGPU_MEM(int) dataDevice = context.Malloc<int>(total);

	IntervalMove(total, gatherDevice->get(), scatterDevice->get(), 
		countsDevice->get(), NumInputs, mgpu::counting_iterator<int>(0),
		dataDevice->get(), context);

	printf("\nMoved data:\n");
	PrintArray(*dataDevice, "%4d", 10);
} 

////////////////////////////////////////////////////////////////////////////////
// Join

void DemoJoin(CudaContext& context) {
	printf("\n\nRELATIONAL JOINS DEMONSTRATION\n\n");

	int ACount = 30;
	int BCount = 30;

	MGPU_MEM(int) aKeysDevice = context.SortRandom<int>(ACount, 100, 130);
	MGPU_MEM(int) bKeysDevice = context.SortRandom<int>(BCount, 100, 130);
	std::vector<int> aKeysHost, bKeysHost;
	aKeysDevice->ToHost(aKeysHost);
	bKeysDevice->ToHost(bKeysHost);

	printf("A keys:\n");
	PrintArray(*aKeysDevice, "%4d", 10);

	printf("\nB keys:\n");
	PrintArray(*bKeysDevice, "%4d", 10);

	MGPU_MEM(int) aIndices, bIndices;
	int innerCount = RelationalJoin<MgpuJoinKindInner>(aKeysDevice->get(),
		ACount, bKeysDevice->get(), BCount, &aIndices, &bIndices, context);

	std::vector<int> aHost, bHost;
	aIndices->ToHost(aHost);
	bIndices->ToHost(bHost);

	printf("\nInner-join (%d items):\n", innerCount);
	printf("output   (aIndex, bIndex) : (aKey, bKey)\n");
	printf("----------------------------------------\n");
	for(int i = 0; i < innerCount; ++i)
		printf("%3d      (%6d, %6d) : (%4d, %4d)\n", i, aHost[i], bHost[i],
			aKeysHost[aHost[i]], bKeysHost[bHost[i]]);

	int outerCount = RelationalJoin<MgpuJoinKindOuter>(aKeysDevice->get(),
		ACount, bKeysDevice->get(), BCount, &aIndices, &bIndices, context);

	aIndices->ToHost(aHost);
	bIndices->ToHost(bHost);
	printf("\nOuter-join (%d items):\n", outerCount);
	printf("output   (aIndex, bIndex) : (aKey, bKey)\n");
	printf("----------------------------------------\n");
	for(int i = 0; i < outerCount; ++i) {
		std::string aKey, bKey;
		if(-1 != aHost[i]) aKey = stringprintf("%4d", aKeysHost[aHost[i]]);
		if(-1 != bHost[i]) bKey = stringprintf("%4d", bKeysHost[bHost[i]]);
		printf("%3d      (%6d, %6d) : (%4s, %4s)\n", i, aHost[i], bHost[i],
			(-1 != aHost[i]) ? aKey.c_str() : "---", 
			(-1 != bHost[i]) ? bKey.c_str() : "---");
	}
}

////////////////////////////////////////////////////////////////////////////////
// SetsKeys

void DemoSetsKeys(CudaContext& context) {
	printf("\n\nMULTISET-KEYS DEMONSTRATION:\n\n");

	// Use CudaContext::SortRandom to generate 100 random sorted integers 
	// between 0 and 99.
	int N = 100;
	MGPU_MEM(int) aData = context.SortRandom<int>(N, 0, 99);
	MGPU_MEM(int) bData = context.SortRandom<int>(N, 0, 99);

	printf("A:\n");
	PrintArray(*aData, "%4d", 10);
	printf("\nB:\n\n");
	PrintArray(*bData, "%4d", 10);
	
	MGPU_MEM(int) intersectionDevice;
	SetOpKeys<MgpuSetOpIntersection, true>(aData->get(), N, bData->get(), N,
		&intersectionDevice, context, false);

	printf("\nIntersection:\n");
	PrintArray(*intersectionDevice, "%4d", 10);

	MGPU_MEM(int) symDiffDevice;
	SetOpKeys<MgpuSetOpSymDiff, true>(aData->get(), N, bData->get(), N,
		&symDiffDevice, context, false);

	printf("\nSymmetric difference:\n");
	PrintArray(*symDiffDevice, "%4d", 10);
}

////////////////////////////////////////////////////////////////////////////////
// SetsPairs

void DemoSetsPairs(CudaContext& context) {
	printf("\n\nMULTISET-PAIRS DEMONSTRATION:\n\n");

	// Use CudaContext::SortRandom to generate 100 random sorted integers 
	// between 0 and 99.
	int N = 100;
	MGPU_MEM(int) aData = context.SortRandom<int>(N, 0, 99);
	MGPU_MEM(int) bData = context.SortRandom<int>(N, 0, 99);

	printf("A:\n");
	PrintArray(*aData, "%4d", 10);
	printf("\nB:\n\n");
	PrintArray(*bData, "%4d", 10);
	
	MGPU_MEM(int) intersectionDevice, intersectionValues;
	SetOpPairs<MgpuSetOpIntersection, true>(aData->get(), 
		mgpu::counting_iterator<int>(0), N, bData->get(), 
		mgpu::counting_iterator<int>(N), N, &intersectionDevice,
		&intersectionValues, context);

	printf("\nIntersection keys:\n");
	PrintArray(*intersectionDevice, "%4d", 10);

	printf("\nIntersection indices:\n");
	PrintArray(*intersectionValues, "%4d", 10);

	MGPU_MEM(int) symDiffDevice, symDiffValues;
	SetOpPairs<MgpuSetOpSymDiff, true>(aData->get(), 
		mgpu::counting_iterator<int>(0), N, bData->get(), 
		mgpu::counting_iterator<int>(N), N, &symDiffDevice, &symDiffValues, 
		context);

	printf("\nSymmetric difference keys:\n");
	PrintArray(*symDiffDevice, "%4d", 10);

	printf("\nSymmetric difference indices:\n");
	PrintArray(*symDiffValues, "%4d", 10);
}

////////////////////////////////////////////////////////////////////////////////
// ReduceByKey

void DemoReduceByKey(CudaContext& context) {
	printf("\n\nREDUCE BY KEY DEMONSTRATION\n\n");

	int count = 100;
	std::vector<int> keys(count);
	for(int i = 1; i < count; ++i)
		keys[i] = keys[i - 1] + (0 == Rand(0, 9));

	MGPU_MEM(int) keysDevice = context.Malloc(keys);
	MGPU_MEM(int) valsDevice = context.GenRandom<int>(count, 1, 5);

	printf("Keys:\n");
	PrintArray(*keysDevice, "%4d", 10);

	printf("\nValues:\n");
	PrintArray(*valsDevice, "%4d", 10);

	MGPU_MEM(int) keysDestDevice = context.Malloc<int>(count);
	MGPU_MEM(int) destDevice = context.Malloc<int>(count);

	int numSegments;
	ReduceByKey(keysDevice->get(), valsDevice->get(), count,
		0, mgpu::plus<int>(), mgpu::equal_to<int>(), destDevice->get(),
		keysDestDevice->get(), &numSegments, (int*)0, context);

	printf("\nReduced keys:\n");
	PrintArray(*keysDestDevice, numSegments, "%4d", 10);

	printf("\nReduced values:\n");
	PrintArray(*destDevice, numSegments, "%4d", 10);
}

////////////////////////////////////////////////////////////////////////////////
// DemoSegReduceCsr

void DemoSegReduceCsr(CudaContext& context) {
	printf("\n\nSEGMENTED REDUCE-CSR DEMONSTRATION\n\n");

	int count = 100;
	const int SegmentStarts[] = {
		0, 9, 19, 25, 71, 87, 97
	};
	const int NumSegments = sizeof(SegmentStarts) / sizeof(int);
	MGPU_MEM(int) csrDevice = context.Malloc(SegmentStarts, NumSegments);
	MGPU_MEM(int) valsDevice = context.GenRandom<int>(count, 1, 5);

	printf("Segment starts (CSR):\n");
	PrintArray(*csrDevice, "%4d", 10);

	printf("\nValues:\n");
	PrintArray(*valsDevice, "%4d", 10);

	MGPU_MEM(int) resultsDevice = context.Malloc<int>(NumSegments);
	SegReduceCsr(valsDevice->get(), csrDevice->get(), count, NumSegments,
		false, resultsDevice->get(), (int)0, mgpu::plus<int>(), context);

	printf("\nReduced values:\n");
	PrintArray(*resultsDevice, "%4d", 10);
}

int main(int argc, char** argv) {
	// Initialize a CUDA device on the default stream.
	ContextPtr context = CreateCudaDevice(argc, argv, true);

	DemoScan(*context);
	DemoMaxReduce(*context);

	DemoBulkRemove(*context);
	DemoBulkInsert(*context);

	DemoMergeKeys(*context);
	DemoMergePairs(*context);

	DemoSortKeys(*context);
	DemoSortPairs(*context);

	DemoSegSortKeys(*context);
	DemoSegSortPairs(*context);
	
	DemoLocalitySortKeys(*context);
	DemoLocalitySortPairs(*context);
	
	DemoSortedSearch(*context);
	DemoSortedSearch2(*context);

	DemoLBS(*context);

	DemoIntervalExpand(*context);
	DemoIntervalMove(*context);

	DemoJoin(*context);

	DemoSetsKeys(*context);
	DemoSetsPairs(*context);

	DemoReduceByKey(*context);
	DemoSegReduceCsr(*context);
	
	return 0;
}
