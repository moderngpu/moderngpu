
include ../common.mk

all: build

build: parallelmerge

parallelmerge.o: parallelmerge.cu 
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpucontext.o: ../src/mgpucontext.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpuutil.o: ../src/mgpuutil.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

parallelmerge: parallelmerge.o mgpucontext.o mgpuutil.o
	nvcc $(NVCCFLAGS) -o $@ $+

run: build
	./parallelmerge

clean:
	rm -f parallelmerge.o mgpucontext.o mgpuutil.o parallelmerge

clobber:
	clean

