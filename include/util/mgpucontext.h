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

#pragma once

#include <cuda_runtime.h>
#include <list>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <memory>

#include "format.h"
#include "util.h"

namespace mgpu {
	
class CudaException : public std::exception {
public:
	cudaError_t error;
	
	CudaException() throw() { }
	CudaException(cudaError_t e) throw() : error(e) { }
	CudaException(const CudaException& e) throw() : error(e.error) { }

	virtual const char* what() const throw() {
		return "CUDA runtime error";
	}
};

class CudaEvent;
class CudaTimer;
class CudaAlloc;
class CudaAllocSimple;
class CudaAllocBuckets;
class CudaDevice;
class CudaContext;

typedef intrusive_ptr<CudaAlloc> AllocPtr;
typedef intrusive_ptr<CudaDevice> DevicePtr;
typedef intrusive_ptr<CudaContext> ContextPtr;
#define MGPU_MEM(type) mgpu::intrusive_ptr< mgpu::CudaDeviceMem< type > >  


////////////////////////////////////////////////////////////////////////////////
// CudaEvent and CudaTimer. 
// Exception-safe wrappers around cudaEvent_t.

class CudaEvent : public noncopyable {
public:
	CudaEvent() { 
		cudaError_t error = cudaEventCreate(&_event);
		if(cudaSuccess != error) throw CudaException(error);
	}
	~CudaEvent() {
		cudaEventDestroy(_event);
	}
	operator cudaEvent_t() { return _event; }
	void Swap(CudaEvent& rhs) {
		std::swap(_event, rhs._event);
	}
private:
	cudaEvent_t _event;
};

class CudaTimer : noncopyable {
	CudaEvent start, end;
public:
	void Start();
	double Split();
	double Throughput(int count, int numIterations);
};


////////////////////////////////////////////////////////////////////////////////
// Customizable allocator.

// CudaAlloc is the interface class all allocator accesses. Users may derive
// this, implement custom allocators, and set it to the device with 
// CudaDevice::SetAllocator.

class CudaAlloc : public CudaBase {
public:
	virtual cudaError_t Malloc(size_t size, void** p) = 0;
	virtual bool Free(void* p) = 0;
	virtual ~CudaAlloc() { }

	const CudaDevice& Device() const { return *_device; }
	CudaDevice& Device() { return *_device; }
protected:
	CudaAlloc(CudaDevice* device) : _device(device) { }
	DevicePtr _device;
};

// A concrete class allocator that simply calls cudaMalloc and cudaFree.
class CudaAllocSimple : public CudaAlloc {
public:
	virtual cudaError_t Malloc(size_t size, void** p);
	virtual bool Free(void* p);
	virtual ~CudaAllocSimple() { }

	CudaAllocSimple(CudaDevice* device) : CudaAlloc(device) { }
};

// A concrete class allocator that uses exponentially-spaced buckets and an LRU
// to reuse allocations. This is the default allocator. It is shared between
// all contexts on the device.
class CudaAllocBuckets : public CudaAlloc {
public:
	CudaAllocBuckets(CudaDevice* device);
	virtual ~CudaAllocBuckets();

	virtual cudaError_t Malloc(size_t size, void** p);
	virtual bool Free(void* p);

	size_t Allocated() const { return _allocated; }
	size_t Committed() const { return _committed; }
	size_t Capacity() const { return _capacity; }

	void Clear();

	void SetCapacity(size_t capacity) {
		_capacity = capacity;
		Compact(0);
	}

private:
	static const int NumBuckets = 122;
	static const size_t BucketSizes[NumBuckets];

	struct MemNode;
	typedef std::list<MemNode> MemList;
	typedef std::map<void*, MemList::iterator> AddressMap;
	typedef std::multimap<int, MemList::iterator> PriorityMap;

	struct MemNode {
		AddressMap::iterator address;
		PriorityMap::iterator priority;
		int bucket;
	};

	void Compact(size_t extra);
	void FreeNode(MemList::iterator memIt);
	int LocateBucket(size_t size) const;

	AddressMap _addressMap;
	PriorityMap _priorityMap;
	MemList _memLists[NumBuckets + 1];

	size_t _capacity, _allocated, _committed;
	int _counter;
};

////////////////////////////////////////////////////////////////////////////////
// CudaDeviceMem

template<typename T>
class CudaDeviceMem : public CudaBase {
	friend class CudaMemSupport;
public:
	~CudaDeviceMem();

	const T* get() const { return _p; }
	T* get() { return _p; }

	operator const T*() const { return get(); }
	operator T*() { return get(); }

	// Size is in units of T, not bytes.
	size_t Size() const { return _size; }

	// Copy from this to the argument array.
	cudaError_t ToDevice(T* data, size_t count) const;
	cudaError_t ToDevice(size_t srcOffest, size_t bytes, void* data) const;
	cudaError_t ToHost(T* data, size_t count) const;
	cudaError_t ToHost(std::vector<T>& data) const;
	cudaError_t ToHost(size_t srcOffset, size_t bytes, void* data) const;

	// Copy from the argument array to this.
	cudaError_t FromDevice(const T* data, size_t count);
	cudaError_t FromDevice(size_t dstOffset, size_t bytes, const void* data);
	cudaError_t FromHost(const std::vector<T>& data);
	cudaError_t FromHost(const T* data, size_t count);
	cudaError_t FromHost(size_t destOffset, size_t bytes, const void* data);

private:
	friend class CudaContext;
	CudaDeviceMem(CudaAlloc* alloc) : _p(0), _size(0), _alloc(alloc) { }

	AllocPtr _alloc;
	T* _p; 
	size_t _size;
};

////////////////////////////////////////////////////////////////////////////////
// CudaMemSupport includes methods for allocating and de-allocating memory.
// This is inherited by CudaContext.

class CudaMemSupport : public CudaBase {
	friend class CudaDevice;
	friend class CudaContext;
public:
	const CudaDevice& Device() const { return _alloc->Device(); }
	CudaDevice& Device() { return _alloc->Device(); }

	void SetAllocator(CudaAlloc* alloc) { _alloc.reset(alloc); }
	CudaAlloc* GetAllocator() { return _alloc.get(); }	

	// Support for creating arrays.
	template<typename T>
	MGPU_MEM(T) Malloc(size_t count);

	template<typename T>
	MGPU_MEM(T) Malloc(const T* data, size_t count);

	template<typename T>
	MGPU_MEM(T) Malloc(const std::vector<T>& data);

	template<typename T>
	MGPU_MEM(T) Fill(size_t count, T fill);

	template<typename T>
	MGPU_MEM(T) FillAscending(size_t count, T first, T step);

	template<typename T>
	MGPU_MEM(T) GenRandom(size_t count, T min, T max);

	template<typename T>
	MGPU_MEM(T) SortRandom(size_t count, T min, T max);

	template<typename T, typename Func>
	MGPU_MEM(T) GenFunc(size_t count, Func f);

protected:
	AllocPtr _alloc;
};

////////////////////////////////////////////////////////////////////////////////
// CudaDevice and CudaContext

ContextPtr CreateCudaDevice(int ordinal);
ContextPtr CreateCudaDevice(int argc, char** argv, bool printInfo = false);

ContextPtr CreateCudaDeviceStream(int ordinal);
ContextPtr CreateCudaDeviceStream(int argc, char** argv, 
	bool printInfo = false);

class CudaDevice : public CudaBase {
	friend ContextPtr CreateCudaDevice(int ordinal);
	friend ContextPtr CreateCudaDeviceStream(int ordinal);

	AllocPtr CreateDefaultAlloc();
	static ContextPtr Create(int ordinal, bool stream);

	CudaDevice() : _ordinal(-1), _compilerVersion(-1) { }
	~CudaDevice() { }
public:
	const cudaDeviceProp& Prop() const { return _prop; }
	int Ordinal() const { return _ordinal; }
	int NumSMs() const { return _prop.multiProcessorCount; }
	
	int ArchVersion() const { return 100 * _prop.major + 10 * _prop.minor; }
	int CompilerVersion() const { return _compilerVersion; }
	void SetCompilerVersion(int ver) { _compilerVersion = ver; }

	std::string DeviceString() const;

	// Set this device as the active device on the thread.
	void SetActive() { cudaSetDevice(_ordinal); }

	// Create a new context on this device. If stream is true a new stream is
	// created, otherwise the context is created on the default stream. The
	// created context attaches to the provided allocator. If alloc is null, 
	// the device creates a new allocator (CudaAllocBuckets) for the context.
	ContextPtr CreateStream(bool stream, CudaAlloc* alloc);

	// Like CreateStream, but the new context uses the provided stream.
	ContextPtr AttachStream(cudaStream_t stream, CudaAlloc* alloc);

private:
	int _ordinal;
	int _compilerVersion;
	cudaDeviceProp _prop;
};

// CudaContext holds a reference to CudaDevice through the allocator it inherits
// from CudaMemSupport.
class CudaContext : public CudaMemSupport {
	friend class CudaDevice;
	CudaContext() : _stream(0) { }
	~CudaContext();
public:
	int NumSMs() const { return Device().NumSMs(); }
	int ArchVersion() const { return Device().ArchVersion(); }
	int CompilerVersion() const { return Device().CompilerVersion(); }
	std::string DeviceString() const { return Device().DeviceString(); }

	cudaStream_t Stream() const { return _stream; }
	void SetStream(cudaStream_t stream) { _stream = stream; }

	// Set this device as the active device on the thread.
	void SetActive() { Device().SetActive(); }

	// Access the included event.
	CudaEvent& Event() { return _event; }

	// Use the included timer.
	CudaTimer& Timer() { return _timer; }
	void Start() { _timer.Start(); }
	double Split() { return _timer.Split(); }
	double Throughput(int count, int numIterations) {
		return _timer.Throughput(count, numIterations);
	}
private:
	cudaStream_t _stream;
	CudaEvent _event;
	CudaTimer _timer;
};


////////////////////////////////////////////////////////////////////////////////
// CudaDeviceMem method implementations

template<typename T>
cudaError_t CudaDeviceMem<T>::ToDevice(T* data, size_t count) const {
	return ToDevice(0, sizeof(T) * count, data);
}
template<typename T>
cudaError_t CudaDeviceMem<T>::ToDevice(size_t srcOffset, size_t bytes, 
	void* data) const {
	cudaError_t error = cudaMemcpy(data, (char*)_p + srcOffset, bytes, 
		cudaMemcpyDeviceToDevice);
	if(cudaSuccess != error) {
		printf("CudaDeviceMem::ToDevice copy error %d\n", error);
		exit(0);
	}
	return error;
}

template<typename T>
cudaError_t CudaDeviceMem<T>::ToHost(T* data, size_t count) const {
	return ToHost(0, sizeof(T) * count, data);
}
template<typename T>
cudaError_t CudaDeviceMem<T>::ToHost(std::vector<T>& data) const {
	data.resize(_size);
	cudaError_t error = cudaSuccess;
	if(_size) error = ToHost(&data[0], _size);
	return error;
}
template<typename T>
cudaError_t CudaDeviceMem<T>::ToHost(size_t srcOffset, size_t bytes, 
	void* data) const {

	cudaError_t error = cudaMemcpy(data, (char*)_p + srcOffset, bytes,
		cudaMemcpyDeviceToHost);
	if(cudaSuccess != error) {
		printf("CudaDeviceMem::ToHost copy error %d\n", error);
		exit(0);
	}
	return error;
}

template<typename T>
cudaError_t CudaDeviceMem<T>::FromDevice(const T* data, size_t count) {
	return FromDevice(0, sizeof(T) * count, data);
}
template<typename T>
cudaError_t CudaDeviceMem<T>::FromDevice(size_t dstOffset, size_t bytes,
	const void* data) {
	if(dstOffset + bytes > sizeof(T) * _size)
		return cudaErrorInvalidValue;
	cudaMemcpy(_p + dstOffset, data, bytes, cudaMemcpyDeviceToDevice);
	return cudaSuccess;
}
template<typename T>
cudaError_t CudaDeviceMem<T>::FromHost(const std::vector<T>& data) {
	cudaError_t error = cudaSuccess;
	if(data.size()) error = FromHost(&data[0], data.size());
	return error;
}
template<typename T>
cudaError_t CudaDeviceMem<T>::FromHost(const T* data, size_t count) {
	return FromHost(0, sizeof(T) * count, data);
}
template<typename T>
cudaError_t CudaDeviceMem<T>::FromHost(size_t dstOffset, size_t bytes,
	const void* data) {
	if(dstOffset + bytes > sizeof(T) * _size)
		return cudaErrorInvalidValue;
	cudaMemcpy(_p + dstOffset, data, bytes, cudaMemcpyHostToDevice);
	return cudaSuccess;
}
template<typename T>
CudaDeviceMem<T>::~CudaDeviceMem() {
	_alloc->Free(_p);
}

////////////////////////////////////////////////////////////////////////////////
// CudaMemSupport method implementations

template<typename T>
MGPU_MEM(T) CudaMemSupport::Malloc(size_t count) {
	MGPU_MEM(T) mem(new CudaDeviceMem<T>(_alloc.get()));
	mem->_size = count;
	cudaError_t error = _alloc->Malloc(sizeof(T) * count, (void**)&mem->_p);
	if(cudaSuccess != error) throw CudaException(cudaErrorMemoryAllocation);
	return mem;
}

template<typename T>
MGPU_MEM(T) CudaMemSupport::Malloc(const T* data, size_t count) {
	MGPU_MEM(T) mem = Malloc<T>(count);
	mem->FromHost(data, count);
	return mem;
}

template<typename T>
MGPU_MEM(T) CudaMemSupport::Malloc(const std::vector<T>& data) {
	MGPU_MEM(T) mem = Malloc<T>(data.size());
	if(data.size()) mem->FromHost(&data[0], data.size());
	return mem;
}

template<typename T>
MGPU_MEM(T) CudaMemSupport::Fill(size_t count, T fill) {
	std::vector<T> data(count, fill);
	return Malloc(data);
}

template<typename T>
MGPU_MEM(T) CudaMemSupport::FillAscending(size_t count, T first, T step) {
	std::vector<T> data(count);
	for(size_t i = 0; i < count; ++i)
		data[i] = first + i * step;
	return Malloc(data);
}

template<typename T>
MGPU_MEM(T) CudaMemSupport::GenRandom(size_t count, T min, T max) {
	std::vector<T> data(count);
	for(size_t i = 0; i < count; ++i)
		data[i] = Rand(min, max);
	return Malloc(data);
}

template<typename T>
MGPU_MEM(T) CudaMemSupport::SortRandom(size_t count, T min, T max) {
	std::vector<T> data(count);
	for(size_t i = 0; i < count; ++i)
		data[i] = Rand(min, max);
	std::sort(data.begin(), data.end());
	return Malloc(data);
}

template<typename T, typename Func>
MGPU_MEM(T) CudaMemSupport::GenFunc(size_t count, Func f) {
	std::vector<T> data(count);
	for(size_t i = 0; i < count; ++i)
		data[i] = f(i);

	MGPU_MEM(T) mem = Malloc<T>(count);
	mem->FromHost(data, count);
	return mem;
}

////////////////////////////////////////////////////////////////////////////////
// Format methods that operate directly on device mem.

template<typename T, typename Op>
std::string FormatArrayOp(const CudaDeviceMem<T>& mem, Op op, int numCols) {
	std::vector<T> host;
	mem.ToHost(host);
	return FormatArrayOp(host, op, numCols);
}

template<typename T>
void PrintArray(const CudaDeviceMem<T>& mem, const char* format, int numCols) {
	std::string s = FormatArrayOp(mem, FormatOpPrintf(format), numCols);
	printf("%s", s.c_str());
}
template<typename T, typename Op>
void PrintArrayOp(const CudaDeviceMem<T>& mem, Op op, int numCols) {
	std::string s = FormatArrayOp(mem, op, numCols);
	printf("%s", s.c_str());
}

} // namespace mgpu
