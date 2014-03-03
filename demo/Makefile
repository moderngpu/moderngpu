
include ../common.mk

all: build

build: demo

mgpucontext.o: ../src/mgpucontext.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpuutil.o: ../src/mgpuutil.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

demo.o: demo.cu
        nvcc $(NVCCFLAGS) -o $@ -c $<

demo: demo.o mgpucontext.o mgpuutil.o
	nvcc $(NVCCFLAGS) -o $@ $+

run: build
	./demo

clean:
	rm -f demo.o mgpucontext.o mgpuutil.o demo

clobber:
	clean

