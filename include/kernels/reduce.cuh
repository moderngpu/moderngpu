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

namespace mgpu {

// Run a high-throughput reduction over multiple CTAs. Used as the upsweep phase
// for global reduce and global scan.
template<typename Tuning, typename InputIt, typename Op>
MGPU_LAUNCH_BOUNDS void KernelReduce(InputIt data_global, int count, 
	int2 task, typename Op::value_type* reduction_global, Op op) {

	typedef MGPU_LAUNCH_PARAMS Params;
	const int NT = Params::NT;
	const int VT = Params::VT;
	const int NV = NT * VT;
	typedef typename Op::input_type input_type;
	typedef typename Op::value_type value_type;
	typedef CTAReduce<NT, Op> R;

	union Shared {
		typename R::Storage reduce;
		input_type inputs[NV];
	};
	__shared__ Shared shared;

	int tid = threadIdx.x;
	int block = blockIdx.x;
	int first = VT * tid;

	int2 range = ComputeTaskRange(block, task, NV, count);

	// total is the sum of encountered elements. It's undefined on the first 
	// loop iteration.
	value_type total = op.Extract(op.Identity(), -1);
	bool totalDefined = false;
	
	// Loop through all tiles returned by ComputeTaskRange.
	while(range.x < range.y) {
		int count2 = min(NV, count - range.x);

		// Read tile data into register.
		input_type inputs[VT];
		DeviceGlobalToReg<NT, VT>(count2, data_global + range.x, tid, inputs);

		if(Op::Commutative) {
			// This path exploits the commutative property of the operator.
			#pragma unroll
			for(int i = 0; i < VT; ++i) {
				int index = NT * i + tid;
				if(index < count2) {
					value_type x = op.Extract(inputs[i], range.x + index);
					total = (i || totalDefined) ? op.Plus(total, x) : x;
				}
			}
		} else {
			// Store the inputs to shared memory and read them back out in
			// thread order.
			DeviceRegToShared<NT, VT>(NV, inputs, tid, shared.inputs);

			value_type x = op.Extract(op.Identity(), -1);			
			#pragma unroll
			for(int i = 0; i < VT; ++i) {
				int index = first + i;
				if(index < count2) {
					value_type y = op.Extract(shared.inputs[index], 
						range.x + index);
					x = i ? op.Plus(x, y) : y;
				}
			}
			__syncthreads();

			// Run a CTA-wide reduction
			x = R::Reduce(tid, x, shared.reduce, op);
			total = totalDefined ? op.Plus(total, x) : x;
		}

		range.x += NV;
		totalDefined = true;
	}  

	if(Op::Commutative)
		// Run a CTA-wide reduction to sum the partials for each thread.
		total = R::Reduce(tid, total, shared.reduce, op);

	if(!tid) reduction_global[block] = total;
}

template<typename InputIt, typename Op>
MGPU_HOST typename Op::value_type Reduce(InputIt data_global, int count, Op op, 
	CudaContext& context) { 

	typedef typename Op::value_type T;
	const int CutOff = 20000;

	T total;
	if(count < CutOff) {
		// Run a single CTA reduction. This requires only a single kernel launch
		// and no atomic work.
		const int NT = 512;
		const int VT = 5;
		typedef LaunchBoxVT<NT, VT> Tuning;
		int2 launch = Tuning::GetLaunchParams(context);
		const int NV = launch.x * launch.y;
		int numTiles = MGPU_DIV_UP(count, NV);
		
		MGPU_MEM(T) reductionDevice = context.Malloc<T>(1);
		KernelReduce<Tuning><<<1, launch.x, 0, context.Stream()>>>(data_global,
			count, make_int2(numTiles, 1), reductionDevice->get(), op);

		reductionDevice->ToHost(0, sizeof(T), &total);
	} else {
		// We have a large reduction so balance over many CTAs. Launch up to 25
		// per CTA for oversubscription.
		const int NT = 128;
		const int VT = 9;
		typedef LaunchBoxVT<NT, VT> Tuning;
		int2 launch = Tuning::GetLaunchParams(context);
		const int NV = launch.x * launch.y;

		int numTiles = MGPU_DIV_UP(count, NV);
		int numBlocks = std::min(context.NumSMs() * 25, numTiles);
		int2 task = DivideTaskRange(numTiles, numBlocks);

		// Reduce on the GPU.
		MGPU_MEM(T) reductionDevice = context.Malloc<T>(numBlocks);
		KernelReduce<Tuning><<<numBlocks, launch.x, 0, context.Stream()>>>(
			data_global, count, task, reductionDevice->get(), op);

		// Copy each CTA reduction to CPU and finish the job.
		std::vector<T> reductionHost;
		reductionDevice->ToHost(reductionHost);

		total = op.Extract(op.Identity(), -1);
		for(int i = 0; i < numBlocks; ++i)
			total = i ? op.Plus(total, reductionHost[i]) : reductionHost[0];
	}
	return total;
}

template<typename InputIt>
MGPU_HOST typename std::iterator_traits<InputIt>::value_type
Reduce(InputIt data_global, int count, CudaContext& context) { 
	typedef typename std::iterator_traits<InputIt>::value_type T;
	return Reduce(data_global, count, ScanOp<ScanOpTypeAdd, T>(), context);
}

} // namespace mgpu
