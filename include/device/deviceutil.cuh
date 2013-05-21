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

#if __CUDA_ARCH__ == 100
	#error "COMPUTE CAPABILITY 1.0 NOT SUPPORTED BY MPGU. TRY 2.0!"
#endif 

#include <climits>
#include "../util/static.h"

namespace mgpu {

#define MGPU_HOST __host__ __forceinline__
#define MGPU_DEVICE __device__ __forceinline__
#define MGPU_HOST_DEVICE __host__ __device__ __forceinline__

const int WARP_SIZE = 32;
const int LOG_WARP_SIZE = 5;

template<typename T>
struct less {
	MGPU_HOST_DEVICE bool operator()(T a, T b) { return a < b; }
};
template<typename T>
struct less_equal {
	MGPU_HOST_DEVICE bool operator()(T a, T b) { return a <= b; }
};
template<typename T>
struct greater {
	MGPU_HOST_DEVICE bool operator()(T a, T b) { return a > b; }
};
template<typename T>
struct greater_equal {
	MGPU_HOST_DEVICE bool operator()(T a, T b) { return a >= b; }
};


template<typename T>
MGPU_HOST_DEVICE void swap(T& a, T& b) {
	T c = a;
	a = b;
	b = c;
}

template<typename T> struct numeric_limits;
template<> struct numeric_limits<int> {
	MGPU_HOST_DEVICE static int min() { return INT_MIN; }
	MGPU_HOST_DEVICE static int max() { return INT_MAX; }
	MGPU_HOST_DEVICE static int lowest() { return INT_MIN; }
};
template<> struct numeric_limits<long long> {
	MGPU_HOST_DEVICE static long long min() { return LLONG_MIN; }
	MGPU_HOST_DEVICE static long long max() { return LLONG_MAX; }
	MGPU_HOST_DEVICE static long long lowest() { return LLONG_MIN; }
};
template<> struct numeric_limits<uint> {
	MGPU_HOST_DEVICE static uint min() { return 0; }
	MGPU_HOST_DEVICE static uint max() { return UINT_MAX; }
	MGPU_HOST_DEVICE static uint lowest() { return 0; }
};
template<> struct numeric_limits<unsigned long long> {
	MGPU_HOST_DEVICE static unsigned long long min() { return 0; }
	MGPU_HOST_DEVICE static unsigned long long max() { return ULLONG_MAX; }
	MGPU_HOST_DEVICE static unsigned long long lowest() { return 0; }
};
template<> struct numeric_limits<float> {
	MGPU_HOST_DEVICE static float min() { return FLT_MIN; }
	MGPU_HOST_DEVICE static float max() { return FLT_MAX; }
	MGPU_HOST_DEVICE static float lowest() { return -FLT_MAX; }
};
template<> struct numeric_limits<double> {
	MGPU_HOST_DEVICE static double min() { return DBL_MIN; }
	MGPU_HOST_DEVICE static double max() { return DBL_MAX; }
	MGPU_HOST_DEVICE static double lowest() { return -DBL_MAX; }
};

template<typename T>
class counting_iterator : public std::iterator_traits<const T*> {
public:
	MGPU_HOST_DEVICE counting_iterator(T value) : _value(value) { }

	MGPU_HOST_DEVICE T operator[](ptrdiff_t i) { 
		return _value + i;
	}
	MGPU_HOST_DEVICE T operator*() {
		return _value;
	}
	MGPU_HOST_DEVICE counting_iterator operator+(ptrdiff_t diff) {
		return counting_iterator(_value + diff);
	}
	MGPU_HOST_DEVICE counting_iterator operator-(ptrdiff_t diff) {
		return counting_iterator(_value - diff);
	}
	MGPU_HOST_DEVICE counting_iterator& operator+=(ptrdiff_t diff) {
		_value += diff;
		return *this;
	}
	MGPU_HOST_DEVICE counting_iterator& operator-=(ptrdiff_t diff) {
		_value -= diff;
		return *this;
	}
private:
	T _value;
};
template<typename T>
class step_iterator : public std::iterator_traits<const T*> {
public:
	MGPU_HOST_DEVICE step_iterator(T base, T step) :
		_base(base), _step(step), _offset(0) { }

	MGPU_HOST_DEVICE T operator[](ptrdiff_t i) { 
		return _base + (_offset + i) * _step; 
	}
	MGPU_HOST_DEVICE T operator*() { 
		return _base + _offset * _step; 
	} 
	MGPU_HOST_DEVICE step_iterator operator+(ptrdiff_t diff) {
		step_iterator it = *this;
		it._offset += diff;
		return it;
	}
	MGPU_HOST_DEVICE step_iterator operator-(ptrdiff_t diff) {
		step_iterator it = *this;
		it._offset -= diff;
		return it;
	}
	MGPU_HOST_DEVICE step_iterator& operator+=(ptrdiff_t diff) { 
		_offset += diff;
		return *this;
	}
	MGPU_HOST_DEVICE step_iterator& operator-=(ptrdiff_t diff) { 
		_offset -= diff;
		return *this;
	}
private:
	ptrdiff_t _offset;
	T _base, _step;	
};

} // namespace mgpu


template<typename T>
MGPU_HOST_DEVICE mgpu::counting_iterator<T> operator+(ptrdiff_t diff,
	mgpu::counting_iterator<T> it) {
	return it + diff;
}
template<typename T>
MGPU_HOST_DEVICE mgpu::counting_iterator<T> operator-(ptrdiff_t diff,
	mgpu::counting_iterator<T> it) {
	return it + (-diff);
}
template<typename T>
MGPU_HOST_DEVICE mgpu::step_iterator<T> operator+(ptrdiff_t diff, 
	mgpu::step_iterator<T> it) {
	return it + diff;
}
template<typename T>
MGPU_HOST_DEVICE mgpu::step_iterator<T> operator-(ptrdiff_t diff, 
	mgpu::step_iterator<T> it) {
	return it + (-diff);
}

namespace mgpu {

// Get the difference between two pointers in bytes.
MGPU_HOST_DEVICE ptrdiff_t PtrDiff(const void* a, const void* b) {
	return (const byte*)b - (const byte*)a;
}

// Offset a pointer by i bytes.
template<typename T> 
MGPU_HOST_DEVICE const T* PtrOffset(const T* p, ptrdiff_t i) {
	return (const T*)((const byte*)p + i);
}
template<typename T>
MGPU_HOST_DEVICE T* PtrOffset(T* p, ptrdiff_t i) {
	return (T*)((byte*)p + i);
}

MGPU_HOST int2 DivideTaskRange(int numItems, int numWorkers) {
	div_t d = div(numItems, numWorkers);
	return make_int2(d.quot, d.rem);
}

MGPU_HOST_DEVICE int2 ComputeTaskRange(int block, int2 task) {
	int2 range;
	range.x = task.x * block;
	range.x += min(block, task.y);
	range.y = range.x + task.x + (block < task.y);
	return range;
}

MGPU_HOST_DEVICE int2 ComputeTaskRange(int block, int2 task, int blockSize, 
	int count) {
	int2 range = ComputeTaskRange(block, task);
	range.x *= blockSize;
	range.y = min(count, range.y * blockSize);
	return range;
}


} // namespace mgpu
