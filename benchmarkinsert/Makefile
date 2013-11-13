
include ../common.mk

all: build

build: benchmarkinsert

benchmarkinsert.o: benchmarkinsert.cu 
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpucontext.o: ../src/mgpucontext.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpuutil.o: ../src/mgpuutil.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

benchmarkinsert: benchmarkinsert.o mgpucontext.o mgpuutil.o
	nvcc $(NVCCFLAGS) -o $@ $+

run: build
	./benchmarkinsert

clean:
	rm -f benchmarkinsert.o mgpucontext.o mgpuutil.o benchmarkinsert

clobber:
	clean

