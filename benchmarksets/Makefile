
include ../common.mk

all: build

build: benchmarksets

benchmarksets.o: benchmarksets.cu 
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpucontext.o: ../src/mgpucontext.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpuutil.o: ../src/mgpuutil.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

benchmarksets: benchmarksets.o mgpucontext.o mgpuutil.o
	nvcc $(NVCCFLAGS) -o $@ $+

run: build
	./benchmarksets

clean:
	rm -f benchmarksets.o mgpucontext.o mgpuutil.o benchmarksets

clobber:
	clean

