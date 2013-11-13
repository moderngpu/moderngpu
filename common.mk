
ifeq ($(dbg),1)
	NVCCFLAGS += -g -G
endif

ifdef NVCC_BITS
	NVCCFLAGS += -m $(NVCC_BITS)
endif

ifdef NVCC_VERBOSE
	NVCCFLAGS += -Xptxas="-v"
endif

INCLUDES := -I ../include

GENCODE_SM20	:= -gencode arch=compute_20,code=sm_20
GENCODE_SM30	:= -gencode arch=compute_30,code=sm_30
GENCODE_SM35	:= -gencode arch=compute_35,code=sm_35

GENCODE_FLAGS	:= $(GENCODE_SM20) $(GENCODE_SM35)

NVCCFLAGS	+= $(GENCODE_FLAGS) $(INCLUDES)
