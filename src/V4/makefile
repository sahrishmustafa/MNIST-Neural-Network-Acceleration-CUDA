# Optimized Makefile for CUDA Neural Network

# CUDA Compiler
NVCC = nvcc

# Check compute capability of the device
COMPUTE_CAPABILITY = $(shell nvcc --help | grep -o "compute_[0-9]*" | sort -r | head -1 | cut -d'_' -f2)
ifeq ($(COMPUTE_CAPABILITY),)
    ARCH = sm_80  # Default to Ampere architecture if detection fails
else ifeq ($(shell test $(COMPUTE_CAPABILITY) -ge 80; echo $$?),0)
    ARCH = sm_80  # Ampere architecture (RTX 30xx)
	endif

NVFLAGS = -arch=$(ARCH) \
          -O3 \
          -lineinfo \
          --use_fast_math \
          --default-stream per-thread \
          -Xcompiler -fopenmp \
          -lcublas

# Debug flags to use when debugging is needed
DEBUG_FLAGS = -g -G -Xcompiler -rdynamic -lineinfo

# Source, executable names, and data directory
SRC = nn.cu
EXE = exe
DATA_DIR = data

# Default rule
all: $(EXE)

# Rule to compile and link
$(EXE): $(SRC)
	$(NVCC) $(NVFLAGS) $< -o $@

# Debug build
debug: $(SRC)
	$(NVCC) $(NVFLAGS) $(DEBUG_FLAGS) $< -o $(EXE)_debug

# Clean command to remove compiled files
clean:
	rm -f $(EXE) $(EXE)_debug *.o

# Run command
run: $(EXE)
	./$(EXE)

# Run with profiling
profile: $(EXE)
	nsys profile --stats=true ./$(EXE)

# Run with detailed memory profiling
memcheck: $(EXE)
	cuda-memcheck ./$(EXE)

# Standard targets
.PHONY: all clean run profile memcheck debug data