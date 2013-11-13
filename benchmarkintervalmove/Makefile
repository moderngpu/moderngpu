
include ../common.mk

all: build

build: benchmarkintervalmove

benchmarkintervalmove.o: benchmarkintervalmove.cu 
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpucontext.o: ../src/mgpucontext.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

mgpuutil.o: ../src/mgpuutil.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

benchmarkintervalmove: benchmarkintervalmove.o mgpucontext.o mgpuutil.o
	nvcc $(NVCCFLAGS) -o $@ $+

run: build
	./benchmarkintervalmove

clean:
	rm -f benchmarkintervalmove.o mgpucontext.o mgpuutil.o benchmarkintervalmove

clobber:
	clean

