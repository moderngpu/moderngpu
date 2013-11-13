
include ../common.mk

all: build

build: benchmarkreducebykey

benchmarkreducebykey.o: benchmarkreducebykey.cu 
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpucontext.o: ../src/mgpucontext.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpuutil.o: ../src/mgpuutil.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

benchmarkreducebykey: benchmarkreducebykey.o mgpucontext.o mgpuutil.o
	nvcc $(NVCCFLAGS) -o $@ $+

run: build
	./benchmarkreducebykey

clean:
	rm -f benchmarkreducebykey.o mgpucontext.o mgpuutil.o benchmarkreducebykey

clobber:
	clean

