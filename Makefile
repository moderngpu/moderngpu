all :
	make -C benchmarkscan -f Makefile
	make -C benchmarkinsert -f Makefile
	make -C benchmarkmerge -f Makefile
	make -C benchmarksort -f Makefile
	make -C benchmarklocalitysort -f Makefile
	make -C benchmarksortedsearch -f Makefile
	make -C benchmarkloadbalance -f Makefile
	make -C benchmarkintervalmove -f Makefile
	make -C benchmarksets -f Makefile
	make -C benchmarkjoin -f Makefile
	make -C benchmarkreducebykey -f Makefile
	make -C benchmarksegreduce -f Makefile

clean :
	make -C benchmarkscan -f Makefile clean
	make -C benchmarkinsert -f Makefile clean
	make -C benchmarkmerge -f Makefile clean
	make -C benchmarksort -f Makefile clean
	make -C benchmarklocalitysort -f Makefile clean
	make -C benchmarksortedsearch -f Makefile clean
	make -C benchmarkloadbalance -f Makefile clean
	make -C benchmarkintervalmove -f Makefile clean
	make -C benchmarksets -f Makefile clean
	make -C benchmarkjoin -f Makefile clean
	make -C benchmarkreducebykey -f Makefile clean
	make -C benchmarksegreduce -f Makefile clean
