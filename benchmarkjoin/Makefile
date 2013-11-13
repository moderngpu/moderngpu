
include ../common.mk

all: build

build: benchmarkjoin

benchmarkjoin.o: benchmarkjoin.cu 
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpucontext.o: ../src/mgpucontext.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpuutil.o: ../src/mgpuutil.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

benchmarkjoin: benchmarkjoin.o mgpucontext.o mgpuutil.o
	nvcc $(NVCCFLAGS) -o $@ $+

run: build
	./benchmarkjoin

clean:
	rm -f benchmarkjoin.o mgpucontext.o mgpuutil.o benchmarkjoin

clobber:
	clean

