
include ../common.mk

all: build

build: benchmarkmerge

benchmarkmerge.o: benchmarkmerge.cu 
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpucontext.o: ../src/mgpucontext.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpuutil.o: ../src/mgpuutil.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

benchmarkmerge: benchmarkmerge.o mgpucontext.o mgpuutil.o
	nvcc $(NVCCFLAGS) -o $@ $+

run: build
	./benchmarkmerge

clean:
	rm -f benchmarkmerge.o mgpucontext.o mgpuutil.o benchmarkmerge

clobber:
	clean

