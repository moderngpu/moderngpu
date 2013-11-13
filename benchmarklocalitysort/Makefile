include ../common.mk

all: build

build: benchmarklocalitysort

benchmarklocalitysort.o: benchmarklocalitysort.cu 
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpucontext.o: ../src/mgpucontext.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpuutil.o: ../src/mgpuutil.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

benchmarklocalitysort: benchmarklocalitysort.o mgpucontext.o mgpuutil.o
	nvcc $(NVCCFLAGS) -o $@ $+

run: build
	./benchmarklocalitysort

clean:
	rm -f benchmarklocalitysort.o mgpucontext.o mgpuutil.o benchmarklocalitysort

clobber:
	clean

