
include ../common.mk

all: build

build: benchmarklaunchbox

benchmarklaunchbox.o: benchmarklaunchbox.cu 
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpucontext.o: ../src/mgpucontext.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpuutil.o: ../src/mgpuutil.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

benchmarklaunchbox: benchmarklaunchbox.o mgpucontext.o mgpuutil.o
	nvcc $(NVCCFLAGS) -o $@ $+

run: build
	./benchmarklaunchbox

clean:
	rm -f benchmarklaunchbox.o mgpucontext.o mgpuutil.o benchmarklaunchbox

clobber:
	clean

