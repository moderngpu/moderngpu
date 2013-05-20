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

#include "../device/deviceutil.cuh"
#include "../device/intrinsics.cuh"

namespace mgpu {

/*
struct ScanOpInterface {
	enum { Commutative }; // true or false
	typedef X input_type;
	typedef Y value_type;
	typedef Z result_type;

	// Extract() takes inputs loaded from global memory and converts to 
	// value_type.
	MGPU_HOST_DEVICE value_type Extract(input_type t, int index);

	// Plus() operates on two value_types. Reduce and Scan do not rely on the 
	// Plus function being commutative - value t1 always represents values that
	// occur earlier in the input stream than t2.
	MGPU_HOST_DEVICE value_type Plus(value_type t1, value_type t2);

	// Combine() prepares a value for storage. Values are combined with the 
	// original input_type element at the same slot. Combine() is not used with
	// Reduce, as Reduce only returns value_types.
	MGPU_HOST_DEVICE result_type Combine(input_type t1, value_type t2);

	// Identity() returns an input_type that interacts benignly with any other
	// value_type in Plus(). The Identity() value_type is always extracted with
	// the index -1. Identity() elements appear at the end of the stream (in the
	// partial last tile) or are returned as the first element for an exclusive
	// scan.
	MGPU_HOST_DEVICE input_type Identity();
};
*/

// Basic scan operators.
enum ScanOpType {
	ScanOpTypeAdd,
	ScanOpTypeMul,
	ScanOpTypeMin,
	ScanOpTypeMax
};

template<ScanOpType OpType, typename T>
struct ScanOp {
	enum { Commutative = true };
	typedef T input_type;
	typedef T value_type;
	typedef T result_type;

	MGPU_HOST_DEVICE value_type Extract(input_type t, int index) { return t; }
	MGPU_HOST_DEVICE value_type Plus(value_type t1, value_type t2) { 
		switch(OpType) {
			case ScanOpTypeAdd: t1 += t2; break;
			case ScanOpTypeMul: t1 *= t2; break;
			case ScanOpTypeMin: t1 = min(t1, t2); break;
			case ScanOpTypeMax: t1 = max(t1, t2); break;
		}
		return t1;
	}
	MGPU_HOST_DEVICE result_type Combine(input_type t1, value_type t2) {
		return t2;
	}
	MGPU_HOST_DEVICE input_type Identity() { return _ident; }

	MGPU_HOST_DEVICE ScanOp(input_type ident) : _ident(ident) { }
	MGPU_HOST_DEVICE ScanOp() {
		switch(OpType) {
			case ScanOpTypeAdd: _ident = 0; break;
			case ScanOpTypeMul: _ident = 1; break;
			case ScanOpTypeMin: _ident = numeric_limits<T>::max(); break;
			case ScanOpTypeMax: _ident = numeric_limits<T>::lowest(); break;
		}
	}

	input_type _ident;
};
typedef ScanOp<ScanOpTypeAdd, int> ScanOpAdd;

// Override the Extract and Combine behavior of the base operator. This prevents
// the Scan kernel from extracting or combining values twice.
template<typename Base>
struct ScanOpValue : public Base {
	typedef typename Base::value_type input_type;
	typedef typename Base::value_type value_type;
	typedef typename Base::value_type result_type;
	MGPU_HOST_DEVICE value_type Extract(value_type t, int index) { return t; }
	MGPU_HOST_DEVICE value_type Combine(value_type t1, value_type t2) {
		return t2; 
	}
	MGPU_HOST_DEVICE value_type Identity() {
		return Base::Extract(Base::Identity(), -1);
	}
	MGPU_HOST_DEVICE ScanOpValue(Base base) : Base(base) { }
};

// Operator to reduce over the count of a bitfield.
struct ScanOpBitCount {
	enum { Commutative = true };
	typedef uint input_type;
	typedef int value_type;
	typedef int result_type;
	MGPU_HOST_DEVICE int Extract(uint t, int index) { return popc(t); }
	MGPU_HOST_DEVICE int Plus(int t1, int t2) { return t1 + t2; }
	MGPU_HOST_DEVICE uint Identity() { return 0; }
	MGPU_HOST_DEVICE int Combine(uint t1, int t2) { return t2; }
};

// Operator to reduce over the count of an array of generic types.
template<typename T>
struct ScanOpTypeCount {
	enum { Commutative = true };
	typedef T input_type;
	typedef int value_type;
	typedef int result_type;
	MGPU_HOST_DEVICE int Extract(T t, int index) { return _zero != t; }
	MGPU_HOST_DEVICE int Plus(int t1, int t2) { return t1 + t2; }
	MGPU_HOST_DEVICE int Combine(T t1, int t2) { return t2; }
	MGPU_HOST_DEVICE T Identity() { return _zero; }
	MGPU_HOST_DEVICE ScanOpTypeCount(T zero) : _zero(zero) { }
	input_type _zero;
};

template<typename T>
struct ScanOpIndex {
	enum { Commutative = false };
	struct Pair { int index; T value; };

	typedef T input_type;
	typedef Pair value_type;
	typedef int result_type;
	MGPU_HOST_DEVICE value_type Extract(T t, int index) {
		Pair p = { index, t };
		return p;
	}
	MGPU_HOST_DEVICE int Combine(T t1, value_type t2) {
		return t2.index;
	}
	MGPU_HOST_DEVICE input_type Identity() { 
		return _identity;
	}
	MGPU_HOST_DEVICE ScanOpIndex(T identity) : _identity(identity) { }
	T _identity;
};

template<typename T>
struct ScanOpMinIndex : ScanOpIndex<T> {
	typedef typename ScanOpIndex<T>::value_type value_type;
	MGPU_HOST_DEVICE value_type Plus(value_type t1, value_type t2) {
		if(t2.value < t1.value) t1 = t2;
		return t1;
	}
	MGPU_HOST_DEVICE ScanOpMinIndex(T max_ = numeric_limits<T>::max()) :
		ScanOpIndex<T>(max_) { }
};
template<typename T>
struct ScanOpMaxIndex : ScanOpIndex<T> {
	typedef typename ScanOpIndex<T>::value_type value_type;
	MGPU_HOST_DEVICE value_type Plus(value_type t1, value_type t2) {
		if(t2.value > t1.value) t1 = t2;
		return t1;
	}
	MGPU_HOST_DEVICE ScanOpMaxIndex(T min_ = numeric_limits<T>::lowest()) :
		ScanOpIndex<T>(min_) { }
};

////////////////////////////////////////////////////////////////////////////////
// CTAReduce

template<int NT, typename Op = ScanOpAdd>
struct CTAReduce {
	typedef typename Op::value_type T;
	enum { Size = NT, Capacity = NT + NT / WARP_SIZE };
	struct Storage { T shared[Capacity]; };

	MGPU_DEVICE static T Reduce(int tid, T x, Storage& storage, Op op = Op()) {
		// Reverse the bits of the source thread ID and make a conflict-free
		// store using a 33-stride spacing.
		int dest = brev(tid)>> (32 - sLogPow2<NT>::value);
		storage.shared[dest + dest / WARP_SIZE] = x;
		__syncthreads();

		// Fold the data in half with each pass.
		int src = tid + tid / WARP_SIZE;
		#pragma unroll
		for(int destCount = NT / 2; destCount >= 1; destCount /= 2) {
			if(tid < destCount) {
				// On the first pass, read this thread's data out of shared 
				// memory.
				if(NT / 2 == destCount) x = storage.shared[src];
				int src2 = destCount + tid;
				x = op.Plus(x, storage.shared[src2 + src2 / WARP_SIZE]);
				storage.shared[src] = x;
			}
			__syncthreads();
		}
		T total = storage.shared[0];
		__syncthreads();
		return total;
	}
};

#if __CUDA_ARCH__ >= 300

template<int NT>
struct CTAReduce<NT, ScanOpAdd> {
	enum { Size = NT, Capacity = WARP_SIZE };
	struct Storage { int shared[Capacity]; };

	MGPU_DEVICE static int Reduce(int tid, int x, Storage& storage, 
		ScanOpAdd op = ScanOpAdd()) {

		const int SecSize = NT / WARP_SIZE;
		int lane = (SecSize - 1) & tid;
		int sec = tid / SecSize;

		#pragma unroll
		for(int offset = 1; offset < SecSize; offset *= 2)
			x = shfl_add(x, offset);

		if(SecSize - 1 == lane) storage.shared[sec] = x;
		__syncthreads();

		if(tid < WARP_SIZE) {
			x = storage.shared[tid];
			#pragma unroll
			for(int offset = 1; offset < WARP_SIZE; offset *= 2)
				x = shfl_add(x, offset);
			storage.shared[tid] = x;
		}
		__syncthreads();

		int reduction = storage.shared[WARP_SIZE - 1];
		__syncthreads();

		return reduction;
	}
};

#endif // __CUDA_ARCH__ >= 300

////////////////////////////////////////////////////////////////////////////////
// CTAScan

template<int NT, typename Op = ScanOpAdd>
struct CTAScan {
	typedef typename Op::value_type T;
	enum { Size = NT, Capacity = 2 * NT + 1 };
	struct Storage { T shared[Capacity]; };

	MGPU_DEVICE static T Scan(int tid, T x, Storage& storage, T* total,
		MgpuScanType type = MgpuScanTypeExc, Op op = Op()) {

		storage.shared[tid] = x;
		int first = 0;
		__syncthreads();

		#pragma unroll
		for(int offset = 1; offset < NT; offset += offset) {
			if(tid >= offset)
				x = op.Plus(storage.shared[first + tid - offset], x);
			first = NT - first;
			storage.shared[first + tid] = x;
			__syncthreads();
		}
		*total = storage.shared[first + NT - 1];
		if(MgpuScanTypeExc == type) 
			x = tid ? 
				storage.shared[first + tid - 1] : 
				op.Extract(op.Identity(), -1);
		__syncthreads();

		return x;
	}
	MGPU_DEVICE static T Scan(int tid, T x, Storage& storage) {
		T total;
		return Scan(tid, x, storage, &total, MgpuScanTypeExc, Op());
	}
};

////////////////////////////////////////////////////////////////////////////////
// Special partial specialization for CTAScan<NT, ScanOpAdd> on Kepler.
// This uses the shfl intrinsic to reduce scan latency.

#if __CUDA_ARCH__ >= 300

template<int NT>
struct CTAScan<NT, ScanOpAdd> {
	enum { Size = NT, NumSegments = WARP_SIZE, SegSize = NT / NumSegments };
	enum { Capacity = NumSegments + 1 };
	struct Storage { int shared[Capacity + 1]; };

	MGPU_DEVICE static int Scan(int tid, int x, Storage& storage, int* total,
		MgpuScanType type = MgpuScanTypeExc, ScanOpAdd op = ScanOpAdd()) {
	
		// Define WARP_SIZE segments that are NT / WARP_SIZE large.
		// Each warp makes log(SegSize) shfl_add calls.
		// The spine makes log(WARP_SIZE) shfl_add calls.
		int lane = (SegSize - 1) & tid;
		int segment = tid / SegSize;

		// Scan each segment using shfl_add.
		int scan = x;
		#pragma unroll
		for(int offset = 1; offset < SegSize; offset *= 2)
			scan = shfl_add(scan, offset, SegSize);

		// Store the reduction (last element) of each segment into storage.
		if(SegSize - 1 == lane) storage.shared[segment] = scan;
		__syncthreads();

		// Warp 0 does a full shfl warp scan on the partials. The total is
		// stored to shared[NumSegments]. (NumSegments = WARP_SIZE)
		if(tid < NumSegments) {
			int y = storage.shared[tid];
			int scan = y;
			#pragma unroll
			for(int offset = 1; offset < NumSegments; offset *= 2)
				scan = shfl_add(scan, offset, NumSegments);
			storage.shared[tid] = scan - y;
			if(NumSegments - 1 == tid) storage.shared[NumSegments] = scan;
		}
		__syncthreads();

		// Add the scanned partials back in and convert to exclusive scan.
		scan += storage.shared[segment];
		if(MgpuScanTypeExc == type) scan -= x;
		*total = storage.shared[NumSegments];
		__syncthreads();

		return scan;
	}
	MGPU_DEVICE static int Scan(int tid, int x, Storage& storage) {
		int total;
		return Scan(tid, x, storage, &total, MgpuScanTypeExc, ScanOpAdd());
	}
};

#endif // __CUDA_ARCH__ >= 300

////////////////////////////////////////////////////////////////////////////////
// CTABinaryScan

template<int NT>
MGPU_DEVICE int CTABinaryScan(int tid, bool x, int* shared, int* total) {
	const int NumWarps = NT / WARP_SIZE;
	int warp = tid / WARP_SIZE;
	int lane = (WARP_SIZE - 1);

	// Store the bit totals for each warp.
	uint bits = __ballot(x);
	shared[warp] = popc(bits);
	__syncthreads();

#if __CUDA_ARCH__ >= 300
	if(tid < NumWarps) { 
		int x = shared[tid];
		int scan = x;
		#pragma unroll
		for(int offset = 1; offset < NumWarps; offset *= 2)
			scan = shfl_add(scan, offset, NumWarps);
		shared[tid] = scan - x;
	}
	__syncthreads();

#else
	// Thread 0 scans warp totals.
	if(!tid) {
		int scan = 0;
		#pragma unroll
		for(int i = 0; i < NumWarps; ++i) {
			int y = shared[i];
			shared[i] = scan;
			scan += y;
		}
		shared[NumWarps] = scan;
	}
	__syncthreads();

#endif // __CUDA_ARCH__ >= 300
	
	// Add the warp scan back into the partials.
	int scan = shared[warp] + __popc(bfe(bits, 0, lane));
	*total = shared[NumWarps];
	__syncthreads();
	return scan;
}

} // namespace mgpu
