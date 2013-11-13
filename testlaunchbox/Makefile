
include ../common.mk

all: build

build: testlaunchbox

testlaunchbox.o: testlaunchbox.cu 
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpucontext.o: ../src/mgpucontext.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpuutil.o: ../src/mgpuutil.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

testlaunchbox: testlaunchbox.o mgpucontext.o mgpuutil.o
	nvcc $(NVCCFLAGS) -o $@ $+

run: build
	./testlaunchbox

clean:
	rm -f testlaunchbox.o mgpucontext.o mgpuutil.o testlaunchbox

clobber:
	clean

