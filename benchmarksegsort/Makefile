
include ../common.mk

all: build

build: benchmarksegsort

benchmarksegsort.o: benchmarksegsort.cu 
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpucontext.o: ../src/mgpucontext.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpuutil.o: ../src/mgpuutil.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

benchmarksegsort: benchmarksegsort.o mgpucontext.o mgpuutil.o
	nvcc $(NVCCFLAGS) -o $@ $+

run: build
	./benchmarksegsort

clean:
	rm -f benchmarksegsort.o mgpucontext.o mgpuutil.o benchmarksegsort

clobber:
	clean

