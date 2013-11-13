
include ../common.mk

all: build

build: benchmarkloadbalance

benchmarkloadbalance.o: benchmarkloadbalance.cu 
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpucontext.o: ../src/mgpucontext.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpuutil.o: ../src/mgpuutil.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

benchmarkloadbalance: benchmarkloadbalance.o mgpucontext.o mgpuutil.o
	nvcc $(NVCCFLAGS) -o $@ $+

run: build
	./benchmarkloadbalance

clean:
	rm -f benchmarkloadbalance.o mgpucontext.o mgpuutil.o benchmarkloadbalance

clobber:
	clean

