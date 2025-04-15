# Compiler
NVCC = nvcc

# Compiler Flags for compilation
NVCCFLAGS = -arch=sm_75 --expt-relaxed-constexpr \
            -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES \
            -O3 --use_fast_math --fmad=true \
            -Xcompiler -fopenmp

# Separate linker flags
LDFLAGS = -lcublas -lcublasLt -lcudart

# Source Files
SRC = nn.cu
OBJ = nn.o
DLINKOBJ = dlink.o
EXE = exe

# Default rule to compile everything
all: $(EXE)

# Rule to compile CUDA source files into object files (separate compilation)
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

# Device linking step (necessary when using -dc)
$(DLINKOBJ): $(OBJ)
	$(NVCC) $(NVCCFLAGS) -dlink $(OBJ) -o $(DLINKOBJ)

# Final executable link step
$(EXE): $(OBJ) $(DLINKOBJ)
	$(NVCC) $(NVCCFLAGS) $(OBJ) $(DLINKOBJ) -o $(EXE) $(LDFLAGS)

# Clean command to remove compiled files
clean:
	rm -f $(OBJ) $(DLINKOBJ) $(EXE) profile.txt

# Profile and run
profile: $(EXE)
	nvprof ./$(EXE) --metrics all --log-file profile.txt

run: $(EXE)
	./$(EXE)
