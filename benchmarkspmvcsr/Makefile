
include ../common.mk

all: build

build: benchmarkspmvcsr

benchmarkspmvcsr.o: benchmarkspmvcsr.cu 
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpucontext.o: ../src/mgpucontext.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpuutil.o: ../src/mgpuutil.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

mmio.o: ../src/mmio.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

sparsematrix.o: ../src/sparsematrix.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

benchmarkspmvcsr: benchmarkspmvcsr.o mgpucontext.o mgpuutil.o mmio.o sparsematrix.o
	nvcc $(NVCCFLAGS) -o $@ $+ -lcusparse

run: build
	./benchmarkspmvcsr

clean:
	rm -f benchmarkspmvcsr.o mgpucontext.o mgpuutil.o benchmarkspmvcsr

clobber:
	clean

