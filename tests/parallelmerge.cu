#include "moderngpu.cuh"

using namespace mgpu;

template<int NT, typename InputIt1, typename InputIt2, typename OutputIt,
	typename Comp>
__global__ void ParallelMergeA(InputIt1 a_global, int aCount, InputIt2 b_global,
	int bCount, OutputIt dest_global, Comp comp) {

	typedef typename std::iterator_traits<InputIt1>::value_type T;

	int gid = threadIdx.x + NT * blockIdx.x;
	if(gid < aCount) {
		T aKey = a_global[gid];
		int lb = BinarySearch<MgpuBoundsLower>(b_global, bCount, aKey, comp);
		dest_global[gid + lb] = aKey;
	}
}

template<int NT, typename InputIt1, typename InputIt2, typename OutputIt,
	typename Comp>
__global__ void ParallelMergeB(InputIt1 a_global, int aCount, InputIt2 b_global,
	int bCount, OutputIt dest_global, Comp comp) {

	typedef typename std::iterator_traits<InputIt2>::value_type T;

	int gid = threadIdx.x + NT * blockIdx.x;
	if(gid < bCount) {
		T bKey = b_global[gid];
		int ub = BinarySearch<MgpuBoundsUpper>(a_global, aCount, bKey, comp);
		dest_global[gid + ub] = bKey;
	}
}


int main(int argc, char** argv) {
	ContextPtr context = CreateCudaDevice(argc, argv, true);

	int ACount = 100;
	int BCount = 100;
	int Count = ACount + BCount;

	MGPU_MEM(int) aData = context->SortRandom<int>(ACount, 0, Count - 1);
	MGPU_MEM(int) bData = context->SortRandom<int>(BCount, 0, Count - 1);
	MGPU_MEM(int) destData = context->Malloc<int>(Count);

	const int NT = 256;
	int aBlocks = MGPU_DIV_UP(ACount, NT);
	int bBlocks = MGPU_DIV_UP(BCount, NT);

	ParallelMergeA<NT><<<aBlocks, NT>>>(aData->get(), ACount, bData->get(),
		BCount, destData->get(), mgpu::less<int>());

	ParallelMergeB<NT><<<bBlocks, NT>>>(aData->get(), ACount, bData->get(),
		BCount, destData->get(), mgpu::less<int>());

	printf("A data:\n");
	PrintArray(*aData, "%4d", 10);

	printf("\nB data:\n");
	PrintArray(*bData, "%4d", 10);

	printf("\nMerged data:\n");
	PrintArray(*destData, "%4d", 10);

	return 0;
}

