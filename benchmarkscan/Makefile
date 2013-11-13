
include ../common.mk

all: build

build: benchmarkscan

benchmarkscan.o: benchmarkscan.cu 
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpucontext.o: ../src/mgpucontext.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpuutil.o: ../src/mgpuutil.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

benchmarkscan: benchmarkscan.o mgpucontext.o mgpuutil.o
	nvcc $(NVCCFLAGS) -o $@ $+

run: build
	./benchmarkscan

clean:
	rm -f benchmarkscan.o mgpucontext.o mgpuutil.o benchmarkscan

clobber:
	clean

