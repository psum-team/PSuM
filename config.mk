# --------------------------------------------------------------------
# Default Configurations 默认配置项目
# --------------------------------------------------------------------

PROJECT_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

# Compiler Settings 编译器设置
CXX      ?= g++
NVCC     ?= nvcc
ACPP     ?= acpp

# 0/1: disable/enable
USE_EIGEN ?= 1
USE_UMFPACK ?= 1
USE_CUDA  ?= 1
CUDA_ARCH ?= sm_75
# umfpack is necessary for cuda_sparselu_gpu

COMMON_FLAGS	= -O3 -std=c++20 -fopenmp=libgomp
NVCCFLAGS 		= -O3 -std=c++20 -Xcompiler -fPIC -arch=$(CUDA_ARCH)

USE_CUDA_SPARSELU_GPU ?= 1
USE_CUDA_SCHURCOMPLEMENT_GPU ?= 1
USE_CUDA_MULTIGRID_GPU ?= 1
USE_EIGEN_SPARSELU_CPU ?= 1
USE_EIGEN_MULTIGRID_CPU ?= 1
USE_EIGEN_SCHURCOMPLEMENT_CPU ?= 1
USE_SYCL_MULTIGRID_GPU ?= 1
USE_SYCL_MULTIGRID32F_GPU ?= 1

# --------------------------------------------------------------------
# Load Local User Configuration (may not exist) 尝试加载本地用户配置
# --------------------------------------------------------------------
-include $(PROJECT_ROOT)/config.mk.local

# --------------------------------------------------------------------
# Set up Compiler and Linker Flags 编译器和链接器设置
# --------------------------------------------------------------------
LIBS = -lstdc++
FIELD_SOLVER_BIN = $(PROJECT_ROOT)/src/field_solver/bin

ifeq ($(USE_CUDA), 1)
    LIBS     += -lcudart -lcusparse -lcublas -lnvrtc
    ifeq ($(USE_CUDA_SPARSELU_GPU), 1)
        PSUM_FIELD_SOLVER_BACKENDS +=  $(FIELD_SOLVER_BIN)/cuda_sparselu_gpu.o
    endif
    ifeq ($(USE_CUDA_SCHURCOMPLEMENT_GPU), 1)
        PSUM_FIELD_SOLVER_BACKENDS +=  $(FIELD_SOLVER_BIN)/cuda_schurcomplement_gpu.o
    endif
    ifeq ($(USE_CUDA_MULTIGRID_GPU), 1)
        PSUM_FIELD_SOLVER_BACKENDS +=  $(FIELD_SOLVER_BIN)/cuda_multigrid_gpu.o
    endif
endif

ifeq ($(USE_UMFPACK), 1)
    LIBS     += -lumfpack
endif

ifeq ($(USE_EIGEN), 1)
    ifeq ($(USE_EIGEN_SPARSELU_CPU), 1)
        PSUM_FIELD_SOLVER_BACKENDS += $(FIELD_SOLVER_BIN)/eigen_sparselu_cpu.o
    endif
	ifeq ($(USE_EIGEN_MULTIGRID_CPU), 1)
        PSUM_FIELD_SOLVER_BACKENDS += $(FIELD_SOLVER_BIN)/eigen_multigrid_cpu.o
    endif
	ifeq ($(USE_EIGEN_SCHURCOMPLEMENT_CPU), 1)
        PSUM_FIELD_SOLVER_BACKENDS += $(FIELD_SOLVER_BIN)/eigen_schurcomplement_cpu.o
    endif
endif

ifeq ($(USE_SYCL_MULTIGRID_GPU), 1)
    PSUM_FIELD_SOLVER_BACKENDS += $(FIELD_SOLVER_BIN)/sycl_multigrid_gpu.o
endif

ifeq ($(USE_SYCL_MULTIGRID32F_GPU), 1)
    PSUM_FIELD_SOLVER_BACKENDS += $(FIELD_SOLVER_BIN)/sycl_multigrid32f_gpu.o
endif

USE_BACKENDS = -Wl,--whole-archive $(FIELD_SOLVER_BIN)/registry.o $(FIELD_SOLVER_BIN)/impls.o -Wl,--no-whole-archive