
include ../common.mk

all: build

build: benchmarksegreduce

benchmarksegreduce.o: benchmarksegreduce.cu 
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpucontext.o: ../src/mgpucontext.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpuutil.o: ../src/mgpuutil.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

benchmarksegreduce: benchmarksegreduce.o mgpucontext.o mgpuutil.o
	nvcc $(NVCCFLAGS) -o $@ $+

run: build
	./benchmarksegreduce

clean:
	rm -f benchmarksegreduce.o mgpucontext.o mgpuutil.o benchmarksegreduce

clobber:
	clean

