#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define INPUT_SIZE   784
#define HIDDEN_SIZE  128
#define OUTPUT_SIZE  10
#define LEARNING_RATE 0.01
#define EPOCHS       3
#define BATCH_SIZE   64
#define NUM_CLASSES  10  // Digits 0-9

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate memory for a 2D matrix (host side)
double** allocateMatrix(int rows, int cols) {
    double** mat = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*)malloc(cols * sizeof(double));
    }
    return mat;
}

// Free allocated matrix memory (host side)
void freeMatrix(double** mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

// Activation functions on host
void relu(double* x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = (x[i] > 0) ? x[i] : 0;
    }
}

void softmax(double* x, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i]);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// -----------------------
// CUDA Kernel Definitions
// -----------------------

// Each thread computes one neuron in the hidden layer.
__global__ void forwardHidden(double* d_W1, double* d_b1, double* d_input, double* d_hidden) {
    int i = threadIdx.x;
    if(i < HIDDEN_SIZE){
        double sum = d_b1[i];
        for (int j = 0; j < INPUT_SIZE; j++){
            sum += d_W1[i * INPUT_SIZE + j] * d_input[j];
        }
        d_hidden[i] = (sum > 0) ? sum : 0;  // ReLU activation
    }
}

// Each thread computes one neuron in the output layer.
__global__ void forwardOutput(double* d_W2, double* d_b2, double* d_hidden, double* d_output) {
    int i = threadIdx.x;
    if(i < OUTPUT_SIZE){
        double sum = d_b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++){
            sum += d_W2[i * HIDDEN_SIZE + j] * d_hidden[j];
        }
        d_output[i] = sum;  // Softmax will be applied on host later
    }
}

// Compute output gradient: d_dOutput = output - target
__global__ void computeOutputGradients(double* d_output, double* d_target, double* d_dOutput) {
    int i = threadIdx.x;
    if(i < OUTPUT_SIZE) {
        d_dOutput[i] = d_output[i] - d_target[i];
    }
}

// Compute hidden gradient: d_dHidden = (W2^T * d_dOutput) * (hidden > 0)
__global__ void computeHiddenGradients(double* d_W2, double* d_dOutput, double* d_hidden, double* d_dHidden) {
    int i = threadIdx.x;
    if(i < HIDDEN_SIZE) {
        double sum = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++){
            sum += d_W2[j * HIDDEN_SIZE + i] * d_dOutput[j];
        }
        d_dHidden[i] = (d_hidden[i] > 0) ? sum : 0.0;
    }
}

// Update output layer weights and biases
// Launch with OUTPUT_SIZE blocks and HIDDEN_SIZE threads per block.
__global__ void updateOutputLayer(double* d_W2, double* d_b2, double* d_dOutput, double* d_hidden) {
    int i = blockIdx.x;      // output neuron index
    int j = threadIdx.x;     // hidden neuron index
    if(i < OUTPUT_SIZE && j < HIDDEN_SIZE) {
        d_W2[i * HIDDEN_SIZE + j] -= LEARNING_RATE * d_dOutput[i] * d_hidden[j];
    }
    if(j == 0 && i < OUTPUT_SIZE) {
        d_b2[i] -= LEARNING_RATE * d_dOutput[i];
    }
}

// Update hidden layer weights and biases
// Launch with HIDDEN_SIZE blocks and INPUT_SIZE threads per block.
__global__ void updateHiddenLayer(double* d_W1, double* d_b1, double* d_dHidden, double* d_input) {
    int i = blockIdx.x;  // hidden neuron index
    int j = threadIdx.x; // input neuron index
    if(i < HIDDEN_SIZE && j < INPUT_SIZE) {
        d_W1[i * INPUT_SIZE + j] -= LEARNING_RATE * d_dHidden[i] * d_input[j];
    }
    if(j == 0 && i < HIDDEN_SIZE) {
        d_b1[i] -= LEARNING_RATE * d_dHidden[i];
    }
}

// -----------------------
// Neural Network Structure
// -----------------------
typedef struct {
    // Device pointers for parameters (flattened arrays)
    double* d_W1; // [HIDDEN_SIZE x INPUT_SIZE]
    double* d_W2; // [OUTPUT_SIZE x HIDDEN_SIZE]
    double* d_b1; // [HIDDEN_SIZE]
    double* d_b2; // [OUTPUT_SIZE]
} NeuralNetwork;

// -----------------------
// Create and Initialize Network (Device Parameters)
// -----------------------
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    
    // Allocate device memory for weights and biases
    cudaMalloc((void**)&net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    cudaMalloc((void**)&net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void**)&net->d_b1, HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void**)&net->d_b2, OUTPUT_SIZE * sizeof(double));
    
    // Allocate temporary host arrays to initialize parameters
    double* h_W1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    double* h_W2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    double* h_b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    double* h_b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));
    
    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++) {
        h_W1[i] = ((double)rand() / RAND_MAX) * 0.01;
    }
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++) {
        h_W2[i] = ((double)rand() / RAND_MAX) * 0.01;
    }
    
    // Copy initialized parameters to device
    cudaMemcpy(net->d_W1, h_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_W2, h_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_b1, h_b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_b2, h_b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    
    free(h_W1); free(h_W2); free(h_b1); free(h_b2);
    
    return net;
}

// -----------------------
// Forward Pass on GPU
// -----------------------
// This function transfers the input vector to the device, runs forward pass kernels,
// copies the output (and hidden layer, needed for backpropagation) back to host,
// and applies the softmax on the output (on the host).
void forwardGPU(NeuralNetwork* net, double* input, double* hidden, double* output) {
    double *d_input, *d_hidden, *d_output;
    size_t size_input = INPUT_SIZE * sizeof(double);
    size_t size_hidden = HIDDEN_SIZE * sizeof(double);
    size_t size_output = OUTPUT_SIZE * sizeof(double);
    
    // Allocate temporary device memory for input, hidden, output vectors
    cudaMalloc((void**)&d_input, size_input);
    cudaMalloc((void**)&d_hidden, size_hidden);
    cudaMalloc((void**)&d_output, size_output);
    
    // Copy host input to device
    cudaMemcpy(d_input, input, size_input, cudaMemcpyHostToDevice);
    
    // Launch forward pass kernels
    // Use one block with HIDDEN_SIZE threads for hidden layer
    forwardHidden<<<1, HIDDEN_SIZE>>>(net->d_W1, net->d_b1, d_input, d_hidden);
    cudaDeviceSynchronize();
    
    // Use one block with OUTPUT_SIZE threads for output layer
    forwardOutput<<<1, OUTPUT_SIZE>>>(net->d_W2, net->d_b2, d_hidden, d_output);
    cudaDeviceSynchronize();
    
    // Copy hidden and output results back to host
    cudaMemcpy(hidden, d_hidden, size_hidden, cudaMemcpyDeviceToHost);
    cudaMemcpy(output, d_output, size_output, cudaMemcpyDeviceToHost);
    
    // Apply softmax on the host
    softmax(output, OUTPUT_SIZE);
    
    // Free temporary device memory
    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);
}

// -----------------------
// Backward Pass on GPU
// -----------------------
// This function assumes that the input, hidden, output activations and target vector are on the host.
// It allocates temporary device memory for these vectors (and gradient vectors), launches the kernels to compute
// gradients and update the parameters (which reside permanently on the device), then frees the temporary memory.
void backwardGPU(NeuralNetwork* net, double* input, double* hidden, double* output, double* target) {
    double *d_input, *d_hidden, *d_output, *d_target;
    double *d_dOutput, *d_dHidden;
    size_t size_input = INPUT_SIZE * sizeof(double);
    size_t size_hidden = HIDDEN_SIZE * sizeof(double);
    size_t size_output = OUTPUT_SIZE * sizeof(double);
    
    // Allocate temporary device memory for input, hidden, output, target, and gradients
    cudaMalloc((void**)&d_input, size_input);
    cudaMalloc((void**)&d_hidden, size_hidden);
    cudaMalloc((void**)&d_output, size_output);
    cudaMalloc((void**)&d_target, size_output);
    cudaMalloc((void**)&d_dOutput, size_output);
    cudaMalloc((void**)&d_dHidden, size_hidden);
    
    // Copy host data to device
    cudaMemcpy(d_input, input, size_input, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hidden, hidden, size_hidden, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, size_output, cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target, size_output, cudaMemcpyHostToDevice);
    
    // Compute output gradients: d_dOutput = output - target
    computeOutputGradients<<<1, OUTPUT_SIZE>>>(d_output, d_target, d_dOutput);
    cudaDeviceSynchronize();
    
    // Compute hidden gradients: d_dHidden = (W2^T * d_dOutput) * (hidden > 0)
    computeHiddenGradients<<<1, HIDDEN_SIZE>>>(net->d_W2, d_dOutput, d_hidden, d_dHidden);
    cudaDeviceSynchronize();
    
    // Update output layer parameters: weights and biases
    // Launch OUTPUT_SIZE blocks with HIDDEN_SIZE threads each.
    updateOutputLayer<<<OUTPUT_SIZE, HIDDEN_SIZE>>>(net->d_W2, net->d_b2, d_dOutput, d_hidden);
    cudaDeviceSynchronize();
    
    // Update hidden layer parameters
    // Launch HIDDEN_SIZE blocks with INPUT_SIZE threads each.
    updateHiddenLayer<<<HIDDEN_SIZE, INPUT_SIZE>>>(net->d_W1, net->d_b1, d_dHidden, d_input);
    cudaDeviceSynchronize();
    
    // Free temporary device memory used for backward pass
    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_target);
    cudaFree(d_dOutput);
    cudaFree(d_dHidden);
}

// -----------------------
// Training and Evaluation (Host Side)
// -----------------------
void train(NeuralNetwork* net, double** images, double** labels, int numImages) {
    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;
        
        for (int i = 0; i < numImages; i++) {
            double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
            // Forward pass on GPU
            forwardGPU(net, images[i], hidden, output);
            
            // Backward pass on GPU: update network parameters (in device memory)
            backwardGPU(net, images[i], hidden, output, labels[i]);
            
            // Compute loss and accuracy on host
            double sample_loss = 0.0;
            for (int k = 0; k < OUTPUT_SIZE; k++)
                sample_loss -= labels[i][k] * log(output[k]);
            loss += sample_loss;
            
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (labels[i][j] > labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;
        }
        
        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    printf("Total training time: %.3fs\n", get_time(total_start));
}

void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages) {
    int correct = 0;
    for (int i = 0; i < numImages; i++) {
        double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
        forwardGPU(net, images[i], hidden, output);
        
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }
    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);
}

// -----------------------
// Data Loading Functions (Unchanged)
// -----------------------
double** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    double** images = allocateMatrix(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }
            images[i][j] = pixel / 255.0;
        }
    }
    fclose(file);
    return images;
}

double** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    double** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }
    fclose(file);
    return labels;
}

// -----------------------
// Free Network Memory (Host Side)
// -----------------------
void freeNetwork(NeuralNetwork* net) {
    // Free device memory
    cudaFree(net->d_W1);
    cudaFree(net->d_W2);
    cudaFree(net->d_b1);
    cudaFree(net->d_b2);
    free(net);
}

// -----------------------
// Main Function
// -----------------------
int main() {
    printf("MNIST Neural Network with CUDA Parameter Allocation\n\n");
    
    double** train_images = loadMNISTImages("data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("data/train-labels.idx1-ubyte", 60000);
    double** test_images  = loadMNISTImages("data/t10k-images.idx3-ubyte", 10000);
    double** test_labels  = loadMNISTLabels("data/t10k-labels.idx1-ubyte", 10000);
    
    NeuralNetwork* net = createNetwork();
    
    train(net, train_images, train_labels, 60000);
    evaluate(net, test_images, test_labels, 10000);
    
    freeNetwork(net);
    
    return 0;
}
