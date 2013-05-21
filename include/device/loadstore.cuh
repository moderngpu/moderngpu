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

#include "../mgpudevice.cuh"
#include "../device/deviceutil.cuh"
#include "../device/intrinsics.cuh"

namespace mgpu {

////////////////////////////////////////////////////////////////////////////////
// Cooperative load functions.

template<int NT, int VT, typename InputIt, typename T>
MGPU_DEVICE void DeviceSharedToReg(int count, InputIt data, int tid, 
	T* reg, bool sync) {

	if(count >= NT * VT) {
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			reg[i] = data[NT * i + tid];
	} else {
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			int index = NT * i + tid;
			if(index < count) reg[i] = data[index];
		}
	}
	if(sync) __syncthreads();
}
template<int NT, int VT, typename InputIt, typename T>
MGPU_DEVICE void DeviceGlobalToReg(int count, InputIt data, int tid, 
	T* reg, bool sync) {
	DeviceSharedToReg<NT, VT>(count, data, tid, reg, sync);
}

////////////////////////////////////////////////////////////////////////////////
// Cooperative store functions.

template<int NT, int VT, typename OutputIt, typename T>
MGPU_DEVICE void DeviceRegToShared(int count, const T* reg, int tid,
	OutputIt dest, bool sync) {
	
	typedef typename std::iterator_traits<OutputIt>::value_type T2;
	if(count >= NT * VT) {
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			dest[NT * i + tid] = (T2)reg[i];
	} else {
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			int index = NT * i + tid;
			if(index < count) 
				dest[index] = (T2)reg[i];
		}
	}
	if(sync) __syncthreads();
}

template<int NT, int VT, typename OutputIt, typename T>
MGPU_DEVICE void DeviceRegToGlobal(int count, const T* reg, int tid,
	OutputIt dest, bool sync) {

	#pragma unroll
	for(int i = 0; i < VT; ++i) {
		int index = NT * i + tid;
		if(index < count)
			dest[index] = reg[i];
	}
	if(sync) __syncthreads();
}

////////////////////////////////////////////////////////////////////////////////
// DeviceMemToMemLoop
// Transfer from shared memory to global, or global to shared, for transfers
// that are smaller than NT * VT in the average case. The goal is to reduce
// unnecessary comparison logic.

template<int NT, int VT, typename InputIt, typename OutputIt>
MGPU_DEVICE void DeviceMemToMem4(int count, InputIt source, int tid,
	OutputIt dest, bool sync) {

	typedef typename std::iterator_traits<InputIt>::value_type T;

	T x[VT];
	const int Count = (VT < 4) ? VT : 4;
	if(count >= NT * VT) {
		#pragma unroll
		for(int i = 0; i < Count; ++i)
			x[i] = source[NT * i + tid];
		#pragma unroll
		for(int i = 0; i < Count; ++i)
			dest[NT * i + tid] = x[i];
	} else {
		#pragma unroll
		for(int i = 0; i < Count; ++i) {
			int index = NT * i + tid;
			if(index < count)
				x[i] = source[NT * i + tid];
		}
		#pragma unroll
		for(int i = 0; i < Count; ++i) {
			int index = NT * i + tid;
			if(index < count)
				dest[index] = x[i];
		}
	}
	if(sync) __syncthreads();
}
template<int NT, typename InputIt, typename OutputIt>
MGPU_DEVICE void DeviceMemToMemLoop(int count, InputIt source, int tid, 
	OutputIt dest, bool sync) {

	for(int i = 0; i < count; i += 4 * NT)
		DeviceMemToMem4<NT, 4>(count - i, source + i, tid, dest + i,
			false);
	if(sync) __syncthreads();
}


////////////////////////////////////////////////////////////////////////////////
// Functions to copy between shared and global memory where the average case is
// to transfer NT * VT elements.

template<int NT, int VT, typename T, typename OutputIt>
MGPU_DEVICE void DeviceSharedToGlobal(int count, const T* source, int tid, 
	OutputIt dest, bool sync) {

	typedef typename std::iterator_traits<OutputIt>::value_type T2;
	#pragma unroll
	for(int i = 0; i < VT; ++i) {
		int index = NT * i + tid;
		if(index < count)
			dest[NT * i + tid] = (T2)source[NT * i + tid];
	}
	if(sync) __syncthreads();
}

template<int NT, int VT, typename InputIt, typename T>
MGPU_DEVICE void DeviceGlobalToShared(int count, InputIt source, int tid,
	T* dest, bool sync) {

	T reg[VT];
	DeviceGlobalToReg<NT, VT>(count, source, tid, reg, false);
	DeviceRegToShared<NT, VT>(NT * VT, reg, tid, dest, sync);
}

template<int NT, int VT, typename InputIt, typename OutputIt>
MGPU_DEVICE void DeviceGlobalToGlobal(int count, InputIt source, int tid, 
	OutputIt dest, bool sync) {

	typedef typename std::iterator_traits<OutputIt>::value_type T;
	T values[VT];
	DeviceGlobalToReg<NT, VT>(count, source, tid, values, false);
	DeviceRegToGlobal<NT, VT>(count, values, tid, dest, sync);
}

////////////////////////////////////////////////////////////////////////////////
// Gather/scatter functions

template<int NT, int VT, typename InputIt, typename T>
MGPU_DEVICE void DeviceGather(int count, InputIt data, int indices[VT], 
	int tid, T* reg, bool sync) {

	if(count >= NT * VT) {
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			reg[i] = data[indices[i]];
	} else {
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			int index = NT * i + tid;
			if(index < count)
				reg[i] = data[indices[i]];
		}
	}
	if(sync) __syncthreads();
}

template<int NT, int VT, typename T>
MGPU_DEVICE void DeviceScatter(int count, const T* reg, int tid, 
	int indices[VT], T* data, bool sync) {

	if(count >= NT * VT) {
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			data[indices[i]] = reg[i];
	} else {
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			int index = NT * i + tid;
			if(index < count)
				data[indices[i]] = reg[i];
		}
	}
	if(sync) __syncthreads();
}

////////////////////////////////////////////////////////////////////////////////
// Cooperative transpose functions (strided to thread order)

template<int VT, typename T>
MGPU_DEVICE void DeviceThreadToShared(const T* threadReg, int tid, T* shared,
	bool sync) {
	#pragma unroll
	for(int i = 0; i < VT; ++i)
		shared[VT * tid + i] = threadReg[i];
	if(sync) __syncthreads();
}

template<int VT, typename T>
MGPU_DEVICE void DeviceSharedToThread(const T* shared, int tid, T* threadReg,
	bool sync) {
	#pragma unroll
	for(int i = 0; i < VT; ++i)
		threadReg[i] = shared[VT * tid + i];
	if(sync) __syncthreads();
}

////////////////////////////////////////////////////////////////////////////////

template<int NT, int VT0, int VT1, typename T>
MGPU_DEVICE void DeviceLoad2ToShared(const T* a_global, int aCount,
	const T* b_global, int bCount, int tid, T* shared, bool sync) {

	int b0 = b_global - a_global - aCount;
	int total = aCount + bCount;
	T reg[VT1];
	if(total >= NT * VT0) {
		#pragma unroll
		for(int i = 0; i < VT0; ++i) {
			int index = NT * i + tid;
			reg[i] = a_global[index + ((index >= aCount) ? b0 : 0)];
		}
	} else {
		#pragma unroll
		for(int i = 0; i < VT0; ++i) {
			int index = NT * i + tid;
			if(index < total)
				reg[i] = a_global[index + ((index >= aCount) ? b0 : 0)];
		}
	}
	#pragma unroll
	for(int i = VT0; i < VT1; ++i) {
		int index = NT * i + tid;
		if(index < total)
			reg[i] = a_global[index + ((index >= aCount) ? b0 : 0)];
	}
	DeviceRegToShared<NT, VT1>(NT * VT1, reg, tid, shared, sync);
}

template<int NT, int VT0, int VT1, typename InputIt1, typename InputIt2,
	typename T>
MGPU_DEVICE void DeviceLoad2ToShared(InputIt1 a_global, int aCount, 
	InputIt2 b_global, int bCount, int tid, T* shared, bool sync) {

	b_global -= aCount;
	int total = aCount + bCount;
	T reg[VT1];
	if(total >= NT * VT0) {
		#pragma unroll
		for(int i = 0; i < VT0; ++i) {
			int index = NT * i + tid;
			if(index < aCount) reg[i] = a_global[index];
			else reg[i] = b_global[index];
		}
	} else {
		#pragma unroll
		for(int i = 0; i < VT0; ++i) {
			int index = NT * i + tid;
			if(index < aCount) reg[i] = a_global[index];
			else if(index < total) reg[i] = b_global[index];
		}
	}
	#pragma unroll
	for(int i = VT0; i < VT1; ++i) {
		int index = NT * i + tid;
		if(index < aCount) reg[i] = a_global[index];
		else if(index < total) reg[i] = b_global[index];
	}
	DeviceRegToShared<NT, VT1>(NT * VT1, reg, tid, shared, sync);
}


////////////////////////////////////////////////////////////////////////////////
// DeviceGatherGlobalToGlobal

template<int NT, int VT, typename InputIt, typename OutputIt>
MGPU_DEVICE void DeviceGatherGlobalToGlobal(int count, InputIt data_global,
	const int* indices_shared, int tid, OutputIt dest_global, bool sync) {

	typedef typename std::iterator_traits<InputIt>::value_type ValType;
	ValType values[VT];

	#pragma unroll
	for(int i = 0; i < VT; ++i) {
		int index = NT * i + tid;
		if(index < count) {
			int gather = indices_shared[index];
			values[i] = data_global[gather];
		}
	}
	if(sync) __syncthreads();
	DeviceRegToGlobal<NT, VT>(count, values, tid, dest_global, false);
}

////////////////////////////////////////////////////////////////////////////////
// DeviceTransferMergeValues
// Gather in a merge-like value from two input arrays and store to a single
// output. Like DeviceGatherGlobalToGlobal, but for two arrays at once.

template<int NT, int VT, typename InputIt1, typename InputIt2,
	typename OutputIt>
MGPU_DEVICE void DeviceTransferMergeValues(int count, InputIt1 a_global, 
	InputIt2 b_global, int bStart, const int* indices_shared, int tid, 
	OutputIt dest_global, bool sync) {

	typedef typename std::iterator_traits<InputIt1>::value_type ValType;
	ValType values[VT];
	
	b_global -= bStart;
	if(count >= NT * VT) {
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			int gather = indices_shared[NT * i + tid];
			values[i] = (gather < bStart) ? a_global[gather] : b_global[gather];
		}	
	} else {
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			int index = NT * i + tid;
			int gather = indices_shared[index];
			if(index < count)
				values[i] = (gather < bStart) ? a_global[gather] :
					b_global[gather];
		}	
	}
	if(sync) __syncthreads();
	DeviceRegToGlobal<NT, VT>(count, values, tid, dest_global, false);
}

template<int NT, int VT, typename T, typename OutputIt>
MGPU_DEVICE void DeviceTransferMergeValues(int count, const T* a_global, 
	const T* b_global, int bStart, const int* indices_shared, int tid, 
	OutputIt dest_global, bool sync) {

	T values[VT];
	int bOffset = (int)(b_global - a_global - bStart);

	if(count >= NT * VT) {
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			int gather = indices_shared[NT * i + tid];
			if(gather >= bStart) gather += bOffset;
			values[i] = a_global[gather];
		}
	} else {
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			int index = NT * i + tid;
			int gather = indices_shared[index];
			if(gather >= bStart) gather += bOffset;
			if(index < count)
				values[i] = a_global[gather];
		}	
	}
	if(sync) __syncthreads();
	DeviceRegToGlobal<NT, VT>(count, values, tid, dest_global, false);
}

} // namespace mgpu
