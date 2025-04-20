# Compiler
CXX = nvc++

# Compiler flags
CXXFLAGS = -acc -Minfo=accel -std=c++11

# Source and target
SRC = nn.cpp
TARGET = exec

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
