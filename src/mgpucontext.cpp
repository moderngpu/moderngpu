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

#include "util/mgpucontext.h"
#include <cassert>
#include <algorithm>
#include <memory>
#include <cstdlib>
#include <cstdio>

namespace mgpu {


////////////////////////////////////////////////////////////////////////////////
// CudaAllocSimple

cudaError_t CudaAllocSimple::Malloc(size_t size, void** p) {
	cudaError_t error = cudaSuccess;
	*p = 0;
	if(size) error = cudaMalloc(p, size);

	if(cudaSuccess != error) {
		printf("CUDA MALLOC ERROR %d\n", error);
		exit(0);
	}

	return error;
}

bool CudaAllocSimple::Free(void* p) {
	cudaError_t error = cudaSuccess;
	if(p) error = cudaFree(p);
	return cudaSuccess == error;
}

////////////////////////////////////////////////////////////////////////////////
// CudaAllocBuckets

CudaAllocBuckets::CudaAllocBuckets(CudaDevice* device) : CudaAlloc(device) { 
	_capacity = _allocated 	= _committed = 0;
	_counter = 0;
}

CudaAllocBuckets::~CudaAllocBuckets() { 
	SetCapacity(0);
	assert(!_allocated);
}

cudaError_t CudaAllocBuckets::Malloc(size_t size, void** p) {

	// Locate the bucket index and adjust the size of the allocation to the 
	// bucket size.
	int bucket = LocateBucket(size);
	if(bucket < NumBuckets)
		size = BucketSizes[bucket];

	// Peel off an already-allocated node and reuse it.
	MemList& list = _memLists[bucket];
	if(list.size() && list.front().priority != _priorityMap.end()) {
		MemList::iterator memIt = list.begin();
		
		_priorityMap.erase(memIt->priority);
		memIt->priority = _priorityMap.end();

		list.splice(list.end(), list, memIt);
		_committed += size;

		*p = memIt->address->first;
		return cudaSuccess;
	}

	// Shrink if this allocation would put us over the limit.
	Compact(size);
	 
	cudaError_t error = cudaSuccess;
	*p = 0;
	if(size) error = cudaMalloc(p, size);
	while((cudaErrorMemoryAllocation == error) && (_committed < _allocated)) {
		SetCapacity(_capacity - _capacity / 10);
		error = cudaMalloc(&p, size);
	}
	if(cudaSuccess != error) return error;

	MemList::iterator memIt = 
		_memLists[bucket].insert(_memLists[bucket].end(), MemNode());
	memIt->bucket = bucket;
	memIt->address = _addressMap.insert(std::make_pair(*p, memIt)).first;
	memIt->priority = _priorityMap.end();
	_allocated += size;
	_committed += size;

	return cudaSuccess;
}

bool CudaAllocBuckets::Free(void* p) {
	AddressMap::iterator it = _addressMap.find(p);
	if(it == _addressMap.end()) {
		// If the pointer was not found in the address map, cudaFree it anyways
		// but return false.
		if(p) cudaFree(p);
		return false;
	}

	// Because we're freeing a page, it had better not be in the priority queue.
	MemList::iterator memIt = it->second;
	assert(memIt->priority == _priorityMap.end());

	// Always free allocations larger than the largest bucket
	if(NumBuckets == memIt->bucket)
		FreeNode(memIt);
	else {
		it->second->priority = _priorityMap.insert(
			std::make_pair(_counter++ - memIt->bucket, memIt));

		// Freed nodes are moved to the front, committed nodes are moved to the
		// end.
		MemList& list = _memLists[memIt->bucket];
		list.splice(list.begin(), list, memIt);
		_committed -= BucketSizes[memIt->bucket];
	}
	Compact(0);
	return true;
}

void CudaAllocBuckets::Clear() {
	Compact(0);
}

void CudaAllocBuckets::FreeNode(CudaAllocBuckets::MemList::iterator memIt) {
	if(memIt->address->first) cudaFree(memIt->address->first);

	_addressMap.erase(memIt->address);
	if(memIt->priority != _priorityMap.end())
		_priorityMap.erase(memIt->priority);
	else
		_committed -= BucketSizes[memIt->bucket];
	_allocated -= BucketSizes[memIt->bucket];

	_memLists[memIt->bucket].erase(memIt);
}

void CudaAllocBuckets::Compact(size_t extra) { 
	while(_allocated + extra > _capacity && _allocated > _committed) {
		// Walk the priority queue from beginning to end removing nodes.
		MemList::iterator memIt = _priorityMap.begin()->second;
		FreeNode(memIt);
	}
}


// Exponentially spaced buckets.
const size_t CudaAllocBuckets::BucketSizes[CudaAllocBuckets::NumBuckets] = {
	       256,        512,       1024,       2048,       4096,       8192,
	     12288,      16384,      24576,      32768,      49152,      65536,
	     98304,     131072,     174848,     218624,     262144,     349696,
	    436992,     524288,     655360,     786432,     917504,    1048576,
	   1310720,    1572864,    1835008,    2097152,    2516736,    2936064,
	   3355648,    3774976,    4194304,    4893440,    5592576,    6291456,
	   6990592,    7689728,    8388608,    9786880,   11184896,   12582912,
	  13981184,   15379200,   16777216,   18874368,   20971520,   23068672,
	  25165824,   27262976,   29360128,   31457280,   33554432,   36910080,
	  40265472,   43620864,   46976256,   50331648,   53687296,   57042688,
	  60398080,   63753472,   67108864,   72701440,   78293760,   83886080,
	  89478656,   95070976,  100663296,  106255872,  111848192,  117440512,
	 123033088,  128625408,  134217728,  143804928,  153391872,  162978816,
	 172565760,  182152704,  191739648,  201326592,  210913792,  220500736,
	 230087680,  239674624,  249261568,  258848512,  268435456,  285212672,
	 301989888,  318767104,  335544320,  352321536,  369098752,  385875968,
	 402653184,  419430400,  436207616,  452984832,  469762048,  486539264,
	 503316480,  520093696,  536870912,  566697216,  596523264,  626349568,
	 656175616,  686001920,  715827968,  745654272,  775480320,  805306368,
	 835132672,  864958720,  894785024,  924611072,  954437376,  984263424,
	1014089728, 1043915776
};

int CudaAllocBuckets::LocateBucket(size_t size) const {
	return (int)(std::lower_bound(BucketSizes, BucketSizes + NumBuckets, size) - 
		BucketSizes);
}


////////////////////////////////////////////////////////////////////////////////
// CudaDevice

ContextPtr CreateCudaDevice(int ordinal) {
	return CudaDevice::Create(ordinal, false);
}
ContextPtr CreateCudaDevice(int argc, char** argv, bool printInfo) {
	int ordinal = 0;
	if(argc >= 2 && !sscanf(argv[1], "%d", &ordinal)) {
		fprintf(stderr, "INVALID COMMAND LINE ARGUMENT - NOT A CUDA ORDINAL\n");
		exit(0);
	}
	ContextPtr context = CreateCudaDevice(ordinal);
	if(printInfo) printf("%s", context->DeviceString().c_str());
	return context;
}

ContextPtr CreateCudaDeviceStream(int ordinal) {
	return CudaDevice::Create(ordinal, true);
}
ContextPtr CreateCudaDeviceStream(int argc, char** argv, bool printInfo) {
	int ordinal = 0;
	if(argc >= 2 && !sscanf(argv[1], "%d", &ordinal)) {
		fprintf(stderr, "INVALID COMMAND LINE ARGUMENT - NOT A CUDA ORDINAL\n");
		exit(0);
	}
	ContextPtr context = CreateCudaDeviceStream(ordinal);
	if(printInfo) printf("%s", context->DeviceString().c_str());
	return context;
}

std::string CudaDevice::DeviceString() const {
	size_t freeMem, totalMem;
	cudaMemGetInfo(&freeMem, &totalMem);
	double memBandwidth = (_prop.memoryClockRate * 1000.0) *
		(_prop.memoryBusWidth / 8 * 2) / 1.0e9;

	std::string s = stringprintf(
		"%s : %8.3lf Mhz   (Ordinal %d)\n"
		"%d SMs enabled. Compute Capability sm_%d%d\n"
		"FreeMem: %6dMB   TotalMem: %6dMB.\n"
		"Mem Clock: %8.3lf Mhz x %d bits   (%1.3lf GB/s)\n"
		"ECC %s\n\n",
		_prop.name, _prop.clockRate / 1000.0, _ordinal,
		_prop.multiProcessorCount, _prop.major, _prop.minor,
		(int)(freeMem / (1<< 20)), (int)(totalMem / (1<< 20)),
		_prop.memoryClockRate / 1000.0, _prop.memoryBusWidth, memBandwidth,
		_prop.ECCEnabled ? "Enabled" : "Disabled");
	return s;
}

AllocPtr CudaDevice::CreateDefaultAlloc() {
	// Create the allocator. Use a bucket allocator with a capacity limit at
	// 80% of free mem.
	intrusive_ptr<CudaAllocBuckets> alloc(new CudaAllocBuckets(this));
	size_t freeMem, totalMem;

	cudaMemGetInfo(&freeMem, &totalMem);
	alloc->SetCapacity((size_t)(.80 * freeMem));
	
	return AllocPtr(alloc.get());
}

ContextPtr CudaDevice::Create(int ordinal, bool stream) {
	// Create the device.
	DevicePtr device(new CudaDevice);
	cudaError_t error = cudaGetDeviceProperties(&device->_prop, ordinal);
	if(cudaSuccess != error) {
		fprintf(stderr, "FAILURE TO CREATE DEVICE %d\n", ordinal);
		exit(0);
	}

	// Set this device as the active one on the thread.
	device->_ordinal = ordinal;
	cudaSetDevice(ordinal);

	AllocPtr alloc = device->CreateDefaultAlloc();

	// Create the context.
	return device->CreateStream(stream, alloc.get());
}

ContextPtr CudaDevice::CreateStream(bool stream, CudaAlloc* alloc) {
	ContextPtr context(new CudaContext);
	context->SetAllocator(alloc ? CreateDefaultAlloc().get() : alloc);

	// Create a stream.
	if(stream) cudaStreamCreate(&context->_stream);
	return context;
}

ContextPtr CudaDevice::AttachStream(cudaStream_t stream, CudaAlloc* alloc) {
	ContextPtr context(new CudaContext);
	context->SetAllocator(alloc ? CreateDefaultAlloc().get() : alloc);

	// Attach the stream.
	context->_stream = stream;
	return context;
}

////////////////////////////////////////////////////////////////////////////////
// CudaContext

CudaContext::~CudaContext() {
	if(_stream) cudaStreamDestroy(_stream);
}

////////////////////////////////////////////////////////////////////////////////
// CudaTimer

void CudaTimer::Start() {
	cudaEventRecord(start);
	cudaDeviceSynchronize();
}
double CudaTimer::Split() {
	cudaEventRecord(end);
	cudaDeviceSynchronize();
	float t;
	cudaEventElapsedTime(&t, start, end);
	start.Swap(end);
	return (t / 1000.0);
}
double CudaTimer::Throughput(int count, int numIterations) {
	double elapsed = Split();
	return (double)numIterations * count / elapsed;
}

} // namespace mgpu
