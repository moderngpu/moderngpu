include ../common.mk

all: build

build: benchmarksortedsearch

benchmarksortedsearch.o: benchmarksortedsearch.cu 
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpucontext.o: ../src/mgpucontext.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpuutil.o: ../src/mgpuutil.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

benchmarksortedsearch: benchmarksortedsearch.o mgpucontext.o mgpuutil.o
	nvcc $(NVCCFLAGS) -o $@ $+

run: build
	./benchmarksortedsearch

clean:
	rm -f benchmarksortedsearch.o mgpucontext.o mgpuutil.o benchmarksortedsearch

clobber:
	clean

