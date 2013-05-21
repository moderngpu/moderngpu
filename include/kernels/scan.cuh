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

#include "../mgpuhost.cuh"
#include "../kernels/reduce.cuh"

namespace mgpu {

////////////////////////////////////////////////////////////////////////////////
// KernelParallelScan
// The "spine" of a global scan.

// Scan inputs on a single CTA. Optionally output the total to dest_global at
// totalIndex.
template<int NT, int VT, MgpuScanType Type, typename InputIt, typename OutputIt, 
	typename Op>
__global__ void KernelParallelScan(InputIt cta_global, int count, Op op, 
	typename Op::value_type* total_global, typename Op::result_type* end_global,
	OutputIt dest_global) {

	typedef typename Op::input_type input_type;
	typedef typename Op::value_type value_type;
	typedef typename Op::result_type result_type;
	const int NV = NT * VT;

	typedef CTAScan<NT, Op> S;
	union Shared {
		typename S::Storage scan;
		input_type inputs[NV];
		result_type results[NV];
	};
	__shared__ Shared shared;

	int tid = threadIdx.x;
	
	// total is the sum of encountered elements. It's undefined on the first 
	// loop iteration.
	value_type total = op.Extract(op.Identity(), -1);
	bool totalDefined = false;
	int start = 0;
	while(start < count) {
		// Load data into shared memory.
		int count2 = min(NV, count - start);
		DeviceGlobalToShared<NT, VT>(count2, cta_global + start, tid, 
			shared.inputs);

		// Transpose data into register in thread order. Reduce terms serially.
		input_type inputs[VT];
		value_type values[VT];
		value_type x = op.Extract(op.Identity(), -1);
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			int index = VT * tid + i;
			if(index < count2) {
				inputs[i] = shared.inputs[index];
				values[i] = op.Extract(inputs[i], start + index);
				x = i ? op.Plus(x, values[i]) : values[i];
			}
		}
		__syncthreads();
				
		// Scan the reduced terms.
		value_type passTotal;
		x = S::Scan(tid, x, shared.scan, &passTotal, MgpuScanTypeExc, op);
		if(totalDefined) {
			x = op.Plus(total, x);
			total = op.Plus(total, passTotal);
		} else
			total = passTotal;

		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			int index = VT * tid + i;
			if(index < count2) {
				// If this is not the first element in the scan, add x values[i]
				// into x. Otherwise initialize x to values[i].
				value_type x2 = (i || tid || totalDefined) ?
					op.Plus(x, values[i]) : 
					values[i];

				// For inclusive scan, set the new value then store.
				// For exclusive scan, store the old value then set the new one.
				if(MgpuScanTypeInc == Type) x = x2;
				shared.results[index] = op.Combine(inputs[i], x);
				if(MgpuScanTypeExc == Type) x = x2;
			}
		}
		__syncthreads();

		DeviceSharedToGlobal<NT, VT>(count2, shared.results, tid, 
			dest_global + start);
		start += NV;
		totalDefined = true;
	}
	
	if(total_global && !tid)
		*total_global = total;

	if(end_global && !tid)
		*end_global = op.Combine(op.Identity(), total);
}

////////////////////////////////////////////////////////////////////////////////
// ScanDownsweepKernel
// Make a second pass through the input array and add in the reduction of the
// preceding values as computed by the spine function ParallelScanKernel.

template<typename Tuning, MgpuScanType Type, typename InputIt,
	typename OutputIt, typename T, typename Op>
MGPU_LAUNCH_BOUNDS void KernelScanDownsweep(InputIt data_global, int count,
	int2 task, const T* reduction_global, OutputIt dest_global, bool totalAtEnd,
	Op op) {

	typedef MGPU_LAUNCH_PARAMS Params;
	const int NT = Params::NT;
	const int VT = Params::VT;
	const int NV = NT * VT;
	typedef typename Op::input_type input_type;
	typedef typename Op::value_type value_type;
	typedef typename Op::result_type result_type;

	typedef CTAScan<NT, Op> S;
	union Shared {
		typename S::Storage scan;
		input_type inputs[NV];
		value_type values[NV];
		result_type results[NV];
	};
	__shared__ Shared shared;

	int tid = threadIdx.x;
	int block = blockIdx.x;
	int2 range = ComputeTaskRange(block, task, NV, count);

	// reduction_global holds the exclusive scan of partials. This is undefined
	// for the first block.
	T next = reduction_global[block];
	bool nextDefined = 0 != block;

	while(range.x < range.y) {
		int count2 = min(NV, count - range.x);

		// Load from global to shared memory.
		DeviceGlobalToShared<NT, VT>(count2, data_global + range.x, tid,
			shared.inputs);

		// Transpose out of shared memory.
		input_type inputs[VT];
		value_type values[VT];
		value_type x = op.Extract(op.Identity(), -1);

		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			int index = VT * tid + i;
			if(index < count2) {
				inputs[i] = shared.inputs[index];
				values[i] = op.Extract(inputs[i], range.x + index);
				x = i ? op.Plus(x, values[i]) : values[i];
			}
		}
		__syncthreads();

		// If nextTotal is defined (i.e. this is not the first frame in the
		// scan), add next into x, then add passTotal into next. Otherwise 
		// set total = passTotal.
		T passTotal;
		x = S::Scan(tid, x, shared.scan, &passTotal, MgpuScanTypeExc, op);
		if(nextDefined) {
			x = op.Plus(next, x);
			next = op.Plus(next, passTotal);
		} else
			next = passTotal;

		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			int index = VT * tid + i;
			if(index < count2) {
				// If this is not the first element in the scan, add x values[i]
				// into x. Otherwise initialize x to values[i].
				value_type x2 = (i || tid || nextDefined) ?
					op.Plus(x, values[i]) : 
					values[i];

				// For inclusive scan, set the new value then store.
				// For exclusive scan, store the old value then set the new one.
				if(MgpuScanTypeInc == Type) x = x2;
				shared.results[index] = op.Combine(inputs[i], x);
				if(MgpuScanTypeExc == Type) x = x2;
			}
		}
		__syncthreads();

		DeviceSharedToGlobal<NT, VT>(count2, shared.results, tid,
			dest_global + range.x);
		range.x += NV;
		nextDefined = true;
	}

	if(totalAtEnd && block == gridDim.x - 1 && !tid)
		dest_global[count] = op.Combine(op.Identity(), next);
}


////////////////////////////////////////////////////////////////////////////////
// Scan host function.
// Uses KernelRakingReduce from kernels/reduce.cuh for upsweep.

template<MgpuScanType Type, typename InputIt, typename OutputIt, typename Op>
MGPU_HOST void Scan(InputIt data_global, int count, OutputIt dest_global, Op op,
	typename Op::value_type* total, bool totalAtEnd, CudaContext& context) {

	typedef typename Op::value_type value_type;
	typedef typename Op::result_type result_type;

	const int CutOff = 20000;
	if(count < CutOff) {
		const int NT = 512;
		const int VT = 3;
		MGPU_MEM(value_type) totalDevice;
		if(total) totalDevice = context.Malloc<value_type>(1);

		KernelParallelScan<NT, VT, Type><<<1, NT, 0, context.Stream()>>>(
			data_global, count, op, total ? totalDevice->get() : (value_type*)0,
			totalAtEnd ? (dest_global + count) : (result_type*)0, dest_global);

		if(total)
			totalDevice->ToHost(total, 1);
	} else {
		
		// Run the parallel raking reduce as an upsweep.
		const int NT = 128;
		const int VT = 7;
		typedef LaunchBoxVT<NT, VT> Tuning;
		int2 launch = Tuning::GetLaunchParams(context);
		const int NV = launch.x * launch.y;

		int numTiles = MGPU_DIV_UP(count, NV);
		int numBlocks = std::min(context.NumSMs() * 25, numTiles);
		int2 task = DivideTaskRange(numTiles, numBlocks);

		MGPU_MEM(value_type) reductionDevice = 
			context.Malloc<value_type>(numBlocks + 1);
		value_type* totalDevice = total ? 
			(reductionDevice->get() + numBlocks) :
			(value_type*)0;
			
		KernelReduce<Tuning><<<numBlocks, launch.x, 0, context.Stream()>>>(
			data_global, count, task, reductionDevice->get(), op);

		// Run a parallel latency-oriented scan to reduce the spine of the 
		// raking reduction.
		const int NT2 = 256;
		const int VT2 = 3;
		KernelParallelScan<NT2, VT2, MgpuScanTypeExc>
			<<<1, NT2, 0, context.Stream()>>>(reductionDevice->get(), numBlocks,
			ScanOpValue<Op>(op), totalDevice, (value_type*)0, 
			reductionDevice->get());
		
		if(total)
			cudaMemcpy(total, totalDevice, sizeof(value_type), 
				cudaMemcpyDeviceToHost);

		// Run a raking scan as a downsweep.
		KernelScanDownsweep<Tuning, Type>
			<<<numBlocks, launch.x, 0, context.Stream()>>>(data_global,
			count, task, reductionDevice->get(), dest_global, totalAtEnd, op);
	}
}

template<MgpuScanType Type, typename InputIt>
MGPU_HOST typename std::iterator_traits<InputIt>::value_type
Scan(InputIt data_global, int count, CudaContext& context) {
	typedef typename std::iterator_traits<InputIt>::value_type T;
	T total;
	Scan<Type>(data_global, count, data_global, ScanOp<ScanOpTypeAdd, T>(), 
		&total, false, context);
	return total;
}
template<typename InputIt>
MGPU_HOST typename std::iterator_traits<InputIt>::value_type
Scan(InputIt data_global, int count, CudaContext& context) {
	return Scan<MgpuScanTypeExc>(data_global, count, context);
}

} // namespace mgpu
