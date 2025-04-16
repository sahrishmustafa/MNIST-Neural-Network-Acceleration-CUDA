# Compiler
NVCC = nvcc

# Compiler Flags
NVFLAGS = -arch=sm_75 

# Source Files
SRC = nn.cu
OBJ = nn.o  
EXE = exe

# Default rule to compile everything
all: $(EXE)

# Rule to compile CUDA source files into object files
%.o: %.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

# Rule to link object files into an executable
$(EXE): $(OBJ)
	$(NVCC) $(NVFLAGS) $(OBJ) -o $(EXE)

# Clean command to remove compiled files
clean:
	rm -f $(OBJ) $(EXE)

profile: $(EXE)
	gprof $(EXE) gmon.out > profile.txt  # Generates profiling report

# Run command to execute the program
run: $(EXE)
	./$(EXE)

