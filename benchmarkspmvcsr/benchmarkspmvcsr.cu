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

#include "sparsematrix.h"
#include "kernels/spmvcsr.cuh"		// MGPU SegReduce Spmv
#include <cusparse_v2.h>			// CUSPARSE

using namespace mgpu;

// HOST FUNCTIONS
// SpmvCsrUnary<T>
// SpmvCsrBinary<T>
// SpmvCsrIndirectUnary<T>
// SpmvCsrIndirectBinary<T>

// SpmvPreprocessUnary<T>
// SpmvPreprocessBinary<T>
// SpmvUnaryApply<T>
// SpmvBinaryApply<T>


enum TestType {
	TestTypeCusparse,
	TestTypeCsr,
	TestTypeCsrPlus
};


const char* Path = "../../matrices/";
struct MatrixDesc {
	const char* filename;
	const char* name;
	bool hasEmpties;
};

const MatrixDesc Matrices[] = {
	{ "dense2.mtx", "Dense", false },
	{ "pdb1HYS.mtx", "Protein", false },
	{ "consph.mtx", "FEM/Spheres", false },
	{ "cant.mtx", "FEM/Cantilever", false },
	{ "pwtk.mtx", "Wind Tunnel", false },
	{ "rma10.mtx", "FEM/Harbor", false },
	{ "qcd5_4.mtx", "QCD", false },
	{ "shipsec1.mtx", "FEM/Ship", false },
	{ "mac_econ_fwd500.mtx", "Economics", false },
	{ "mc2depi.mtx", "Epidemiology", false },
	{ "cop20k_A.mtx", "FEM/Accelerator", true },
	{ "scircuit.mtx", "Circuit", false },
	{ "webbase-1M.mtx", "Webbase", false },
	{ "rail4284.mtx", "LP", false }
};
const int NumMatrices = sizeof(Matrices) / sizeof(*Matrices);

template<typename T>
struct DeviceMatrix {
	int height, width, nz;
	MGPU_MEM(int) csrDevice, colsDevice;
	MGPU_MEM(T) matrixDevice;
};

void BuildDeviceMatrix(const SparseMatrix& m, 
	std::auto_ptr<DeviceMatrix<float> >* ppMatrix, CudaContext& context) {

	std::auto_ptr<DeviceMatrix<float> > m2(new DeviceMatrix<float>);
	m2->height = m.height;
	m2->width = m.width;
	m2->nz = m.nz;
	m2->csrDevice = context.Malloc(m.csr);
	m2->colsDevice = context.Malloc(m.cols);

	// Convert double to float.
	std::vector<float> matrixFloat(m.nz);
	for(int i = 0; i < m.nz; ++i)
		matrixFloat[i] = (float)m.matrix[i];
	m2->matrixDevice = context.Malloc(matrixFloat);

	*ppMatrix = m2;
}

void BuildDeviceMatrix(const SparseMatrix& m, 
	std::auto_ptr<DeviceMatrix<double> >* ppMatrix, CudaContext& context) {

	std::auto_ptr<DeviceMatrix<double> > m2(new DeviceMatrix<double>);

	m2->height = m.height;
	m2->width = m.width;
	m2->nz = m.nz;
	m2->csrDevice = context.Malloc(m.csr);
	m2->colsDevice = context.Malloc(m.cols);
	m2->matrixDevice = context.Malloc(m.matrix);

	*ppMatrix = m2;
}

double CusparseSpmv(int height, int width, int nz, const double* matrix_global,
	const int* csr_global, const int* cols_global, const double* x_global,
	double* y_global, int numIterations, cusparseHandle_t cusparse, 
	CudaContext& context) {

	cusparseMatDescr_t desc;
	cusparseCreateMatDescr(&desc);
	double alpha = 1.0, beta = 0.0;

	context.Start();
	for(int it = 0; it < numIterations; ++it) {
		cusparseStatus_t status = cusparseDcsrmv_v2(cusparse, 
			CUSPARSE_OPERATION_NON_TRANSPOSE, height, width, nz, &alpha,
			desc, matrix_global, csr_global, cols_global, x_global, &beta,
			y_global);
	}
	double elapsed = context.Split();

	cusparseDestroyMatDescr(desc);
	return elapsed;
}

double CusparseSpmv(int height, int width, int nz, const float* matrix_global,
	const int* csr_global, const int* cols_global, const float* x_global,
	float* y_global, int numIterations, cusparseHandle_t cusparse, 
	CudaContext& context) {

	cusparseMatDescr_t desc;
	cusparseCreateMatDescr(&desc);
	float alpha = 1.0, beta = 0.0;

	context.Start();
	for(int it = 0; it < numIterations; ++it) {
		cusparseStatus_t status = cusparseScsrmv_v2(cusparse, 
			CUSPARSE_OPERATION_NON_TRANSPOSE, height, width, nz, &alpha,
			desc, matrix_global, csr_global, cols_global, x_global, &beta,
			y_global);
	}
	double elapsed = context.Split();

	cusparseDestroyMatDescr(desc);
	return elapsed;
}

template<typename T>
double BenchmarkSpmvType(TestType testType, int height, int width, int nz,
	const T* matrix_global, const int* csr_global, const int* cols_global,
	const T* x_global, T* y_global, int numIterations, bool supportEmpty,
	cusparseHandle_t cusparse, CudaContext& context) {

	double elapsed;
	if(TestTypeCusparse == testType) {
		// CUSPARSE
		elapsed = CusparseSpmv(height, width, nz, matrix_global, csr_global,
			cols_global, x_global, y_global, numIterations, cusparse, context);

	} else if(TestTypeCsr == testType) {
		// MGPU CSR
		context.Start();
		for(int it = 0; it < numIterations; ++it)
			SpmvCsrBinary(matrix_global, cols_global, nz, csr_global, height, 
				x_global, supportEmpty, y_global, (T)0, mgpu::multiplies<T>(),
				mgpu::plus<T>(), context);
		elapsed = context.Split();

	} else {
		// MGPU CSR+ (Preprocessed)
		std::auto_ptr<SpmvPreprocessData> preprocessData;
		SpmvPreprocessBinary<T>(nz, csr_global, height, supportEmpty,
			&preprocessData, context);

		context.Start();
		for(int it = 0; it < numIterations; ++it) {
			SpmvBinaryApply(*preprocessData, matrix_global, cols_global,
				x_global, y_global, (T)0, mgpu::multiplies<T>(), 
				mgpu::plus<T>(), context);
		}
		elapsed = context.Split();
	}

	return elapsed;
}

template<typename T>
void BenchmarkSpmv(int test, int numIterations, TestType testType,
	cusparseHandle_t cusparse, CudaContext& context) {

#ifdef _DEBUG
		numIterations = 1;
#endif

	// Load a sparse matrix into host memory.
	std::auto_ptr<SparseMatrix> m;
	std::string err;
	std::string filename = stringprintf("%s%s", Path, Matrices[test].filename);
	bool success = LoadCachedMatrix(filename.c_str(), &m, err);
	if(!success) {
		printf("%s", err.c_str());
		exit(0);
	}

	bool supportEmpty = Matrices[test].hasEmpties;
	
	// Load the sparse matrix into device memory.
	std::auto_ptr<DeviceMatrix<T> > m2;
	BuildDeviceMatrix(*m, &m2, context);

	// Create a vector and fill with random integers.
	std::vector<T> xHost(m->width);
	for(int i = 0; i < m->width; ++i)
		xHost[i] = (T)Rand(1, 9);
	MGPU_MEM(T) xDevice = context.Malloc(xHost);

	// Compute the reference product.
	std::vector<T> yHost(m->height);
	for(int row = 0; row < m->height; ++row) {
		int begin = m->csr[row];
		int end = (row + 1 < m->height) ? m->csr[row + 1] : m->nz;
		T x = 0;
		for(int i = begin; i < end; ++i)
			x += m->matrix[i] * xHost[m->cols[i]];
		yHost[row] = x;
	}
	
	// Benchmark the Spmv.
	MGPU_MEM(T) yDevice = context.Fill<T>(m->height, -1);

	double elapsed = BenchmarkSpmvType<T>(testType, m->height, m->width, m->nz,
		m2->matrixDevice->get(), m2->csrDevice->get(), m2->colsDevice->get(),
		xDevice->get(), yDevice->get(), numIterations, supportEmpty, cusparse, 
		context);

	double mgpuThroughput = (double)numIterations * m->nz / elapsed;
	double mgpuBandwidth = (2 * sizeof(T) + sizeof(int)) * mgpuThroughput;
	printf("%30s %9d NZ %8.3lf GFlops   %8.3lf GB/s\t", Matrices[test].name,
		m->nz, 2 * mgpuThroughput / 1.0e9, mgpuBandwidth / 1.0e9);

	// Check results.
	std::vector<T> resultsHost;
	yDevice->ToHost(resultsHost);
	if(8 == sizeof(T))	// Only compare for doubles.
		CompareVecs(&resultsHost[0], &yHost[0], m->height);

	printf("\n");
}

int main(int argc, char** argv) {
	ContextPtr context = CreateCudaDevice(argc, argv, true);

	cusparseHandle_t cusparse;
	cusparseStatus_t cusparseStatus = cusparseCreate(&cusparse);
	if(CUSPARSE_STATUS_SUCCESS != cusparseStatus) {
		printf("COULD NOT INIT CUSPARSE LIBRARY\n");
		exit(0);
	}

	printf("CSR:\n");
	printf("float:\n");
	for(int test = 0; test < NumMatrices; ++test)
		BenchmarkSpmv<float>(test, 1000, TestTypeCsr, cusparse, *context);
	
	printf("\ndouble:\n");
	for(int test = 0; test < NumMatrices; ++test)
		BenchmarkSpmv<double>(test, 1000, TestTypeCsr, cusparse, *context);
	
	printf("CSR+:\n");
	printf("float:\n");
	for(int test = 0; test < NumMatrices; ++test)
		BenchmarkSpmv<float>(test, 1000, TestTypeCsrPlus, cusparse, *context);

	printf("\ndouble:\n");
	for(int test = 0; test < NumMatrices; ++test)
		BenchmarkSpmv<double>(test, 1000, TestTypeCsrPlus, cusparse, *context);
	
	printf("CUSPARSE:\n");
	printf("float:\n");
	for(int test = 0; test < NumMatrices; ++test)
		BenchmarkSpmv<float>(test, 1000, TestTypeCusparse, cusparse, *context);

	printf("\ndouble:\n");
	for(int test = 0; test < NumMatrices; ++test)
		BenchmarkSpmv<double>(test, 1000, TestTypeCusparse, cusparse, *context);
	
	cusparseDestroy(cusparse);

	return 0;
}

