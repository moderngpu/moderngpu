/* 
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

#include "kernels/segmentedsort.cuh"

using namespace mgpu;

template<typename T>
void TestSegsort(int count, CudaContext& context) {
	// Randomly set flag bits.
	std::vector<uint> flagsHost(MGPU_DIV_UP(count, 32));
	std::vector<int> segmentsHost;
	std::vector<T> keysHost(count);
	for(int i = 0; i < count; ++i) {
		if(0 == (rand() % 32)) {
			flagsHost[i / 32] |= 1<< (31 & i);
			segmentsHost.push_back(i);
		}
		keysHost[i] = (T)rand();
	}

	MGPU_MEM(T) keys = context.Malloc(keysHost);
	MGPU_MEM(T) values = context.FillAscending<T>(count, 0, 1);
	MGPU_MEM(int) segments = context.Malloc(segmentsHost);
	MGPU_MEM(uint) flags = context.Malloc(flagsHost);

	SegSortPairsFromIndices(keys->get(), values->get(), count, 
		segments->get(), (int)segmentsHost.size(), context);
	std::vector<T> keysResults1, valuesResults1;
	keys->ToHost(keysResults1);
	values->ToHost(valuesResults1);

	keys = context.Malloc(keysHost);
	values = context.FillAscending<T>(count, 0, 1);
	
	SegSortPairsFromFlags(keys->get(), values->get(), count, 
		flags->get(), context);
	std::vector<T> keysResults2, valuesResults2;
	keys->ToHost(keysResults2);
	values->ToHost(valuesResults2);

	for(int i = 0; i < count; ++i) {
		if(keysResults1[i] != keysResults2[i]) {
			printf("ERROR AT KEYS[%d]\n", i);
			exit(0);
		}
		if(valuesResults1[i] != valuesResults2[i]) {
			printf("ERROR AT VALUES[%d]\n", i);
			exit(0);
		}
	}
}

int main(int argc, char** argv) {
	ContextPtr context = CreateCudaDevice(argc, argv, true);

	TestSegsort<int>(10000000, *context);
	printf("SUCCESS\n");

	return 0;
}
