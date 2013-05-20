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

#include "util/util.h"

#define MGPU_RAND_NS std::tr1

#ifdef _MSC_VER
#include <random>
#else
#include <tr1/random>
#endif

namespace mgpu {

////////////////////////////////////////////////////////////////////////////////
// Random number generators.

MGPU_RAND_NS::mt19937 mt19937;

int Rand(int min, int max) {
	MGPU_RAND_NS::uniform_int<int> r(min, max);
	return r(mt19937);
}
int64 Rand(int64 min, int64 max) {
	MGPU_RAND_NS::uniform_int<int64> r(min, max);
	return r(mt19937);
}
uint Rand(uint min, uint max) {
	MGPU_RAND_NS::uniform_int<uint> r(min, max);
	return r(mt19937);
}
uint64 Rand(uint64 min, uint64 max) {
	MGPU_RAND_NS::uniform_int<uint64> r(min, max);
	return r(mt19937);
}
float Rand(float min, float max) {
	MGPU_RAND_NS::uniform_real<float> r(min, max);
	return r(mt19937);
}
double Rand(double min, double max) {
	MGPU_RAND_NS::uniform_real<double> r(min, max);
	return r(mt19937);
}

} // namespace mgpu
