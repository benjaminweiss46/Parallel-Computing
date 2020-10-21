
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dataExtractor.h"
#include "gputimer.h"

/**
__global__ void operatorKernel(struct container* data, int threads) {
    int gridSpot = (threadIdx.x + (blockDim.x * blockIdx.x));
    if (gridSpot >= threads) {
        return;
    }
    int A = data[gridSpot].A;
    int B = data[gridSpot].B;
    int op = data[gridSpot].op;
    int result;
    if (op == 0) { // AND
        result = A & B;
    }
    else if (op == 2) { // NAND
        result = !(A * B);
    }
    else if (op == 1) { // OR
        result = A | B;
    }
    else if (op == 3) { // NOR
        result = !(A + B);
    }
    else if (op == 4) {  // XOR
        result = A ^ B;
    }
    else { // XNOR
        result = !(A ^ B);
    }
    data[gridSpot].result = result;
}

int main(int argc, char* argv[]) {
    // parse arguments
    if (argc == 4) {
        printf("Running parallel explicit operators\n");
    }
    else if (argc > 4) {
        printf("Too many arguments given.\n");
        return -1;
    }
    else {
        printf("Missing arguments\n");
        return -1;
    }
    char* inputFile = argv[1];
    char* outputFile = argv[3];
    int size = atoi(argv[2]);

    // create timer
    GpuTimer tim = GpuTimer();

    // get input data
    struct container * data = getCSVData(inputFile, size); //not defensive coding at all but no requirement specified

    // perform logic operations

    //1. Allocate host memory and initialize host data
    struct container * copyData;
    copyData = (struct container *)malloc(sizeof(struct container) * size); 

    //2. Allocate device memory
    struct container* dataGPU;
    cudaMalloc((void**)&dataGPU, (sizeof(struct container) * size));

    //3. Transfer data from host to device memory
    tim.Start();
    cudaMemcpy(dataGPU, data, (sizeof(struct container) * size), cudaMemcpyHostToDevice);
    tim.Stop();
    cudaDeviceSynchronize();
    float t_transfer = tim.Elapsed();

    //4. Execute kernels "go my children, run with haste and do my bidding"
    printf("Sending out threads to do my dirty work\n");
    int blocks;
    if (size > 1024) {
        blocks = (size + (1024 - 1)) / 1024;
    }
    else {
        blocks = 1;
    }
    tim.Start();
    operatorKernel <<<blocks, 1024>>> (dataGPU, size);
    tim.Stop();
    cudaDeviceSynchronize();
    float t_kernel = tim.Elapsed();

    //5. transfer output from device memory to host memory 
    cudaMemcpy(copyData, dataGPU, (sizeof(struct container) * size), cudaMemcpyDeviceToHost);
    cudaFree(dataGPU);
    
    // write results to output file
    char* newline = "\n";
    char buffer[10];
    char* res;
    FILE* solution = fopen(outputFile, "w");
    if (solution == NULL) {
        return -1;
    }
    for (int i = 0; i < size; i++) {
        res = itoa(copyData[i].result, buffer, 10);
        fprintf(solution, res);
        if (!(i == size - 1)) {
            fprintf(solution, newline);
        }
    }
    fclose(solution);

    printf("Explicit data migration time: %f ms\nKernel execution time: %f ms\n", t_transfer, t_kernel);
    return 0;
}
**/