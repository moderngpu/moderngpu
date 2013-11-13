
include ../common.mk

all: build

build: benchmarksort

benchmarksort.o: benchmarksort.cu 
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpucontext.o: ../src/mgpucontext.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpuutil.o: ../src/mgpuutil.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

benchmarksort: benchmarksort.o mgpucontext.o mgpuutil.o
	nvcc $(NVCCFLAGS) -o $@ $+

run: build
	./benchmarksort

clean:
	rm -f benchmarksort.o mgpucontext.o mgpuutil.o benchmarksort

clobber:
	clean

