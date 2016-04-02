
ARCH=-gencode arch=compute_20,code=sm_20 \
 	  -gencode arch=compute_35,code=sm_35 \
  	-gencode arch=compute_52,code=sm_52
# ARCH=-gencode arch=compute_35,code=sm_35

OPTIONS=-std=c++11 -O2 -g -Xcompiler="-Werror" -lineinfo  --expt-extended-lambda -use_fast_math -Xptxas="-v" -I src

all: \
	tests \
	tutorials \
	demos

# kernel tests

expt:
	test_compact

test_compact: tests/test_compact.cu src/moderngpu/*.hxx
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

tests: \
	test_reduce \
	test_scan \
	test_bulkremove \
	test_merge \
	test_bulkinsert \
	test_mergesort \
	test_segsort \
	test_load_balance \
	test_intervalexpand \
	test_intervalmove \
	test_sortedsearch \
	test_join \
	test_segreduce

test_reduce: tests/test_reduce.cu src/moderngpu/*.hxx
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_scan: tests/test_scan.cu src/moderngpu/*.hxx
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_bulkremove: tests/test_bulkremove.cu src/moderngpu/*.hxx
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_merge: tests/test_merge.cu src/moderngpu/*.hxx
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_bulkinsert: tests/test_bulkinsert.cu src/moderngpu/*.hxx
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_mergesort: tests/test_mergesort.cu	src/moderngpu/*.hxx
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_segsort: tests/test_segsort.cu src/moderngpu/*.hxx
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_load_balance: tests/test_load_balance.cu src/moderngpu/*.hxx
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_intervalexpand: tests/test_intervalexpand.cu src/moderngpu/*.hxx
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_intervalmove: tests/test_intervalmove.cu src/moderngpu/*.hxx
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_sortedsearch: tests/test_sortedsearch.cu src/moderngpu/*.hxx
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_join: tests/test_join.cu src/moderngpu/*.hxx
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_segreduce: tests/test_segreduce.cu src/moderngpu/*.hxx
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

# simple tutorials

tutorials: \
	tut_01_transform \
	tut_02_cta_launch \
	tut_03_launch_box \
	tut_04_launch_custom \
	tut_05_iterators \

tut_01_transform: tutorial/tut_01_transform.cu src/moderngpu/*.hxx
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

tut_02_cta_launch: tutorial/tut_02_cta_launch.cu src/moderngpu/*.hxx
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

tut_03_launch_box: tutorial/tut_03_launch_box.cu src/moderngpu/*.hxx
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

tut_04_launch_custom: tutorial/tut_04_launch_custom.cu src/moderngpu/*.hxx
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

tut_05_iterators: tutorial/tut_05_iterators.cu src/moderngpu/*.hxx
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

demos: \
	cities \
	bfs \
	bfs2

cities: demo/cities.cu src/moderngpu/*.hxx
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

bfs: demo/bfs.cu demo/graph.cxx src/moderngpu/*.hxx
	nvcc $(ARCH) $(OPTIONS) -o $@ $< demo/graph.cxx

bfs2: demo/bfs2.cu demo/graph.cxx src/moderngpu/*.hxx
	nvcc $(ARCH) $(OPTIONS) -o $@ $< demo/graph.cxx

clean:
	rm test_reduce
	rm test_scan
	rm test_bulkremove
	rm test_merge
	rm test_bulkinsert
	rm test_mergesort
	rm test_segsort
	rm test_load_balance
	rm test_intervalexpand
	rm test_intervalmove
	rm test_sortedsearch
	rm test_join
	rm test_segreduce
	rm tut_01_transform
	rm tut_02_cta_launch
	rm tut_03_launch_box
	rm tut_04_launch_custom
	rm tut_05_iterators
	rm cities
	rm bfs