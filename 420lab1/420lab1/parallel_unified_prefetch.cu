
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dataExtractor.h"
#include "gputimer.h"


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
    struct container* data = getCSVData(inputFile, size); //not denfensive coding at all but no requirement specifiedS

    // perform logic operations

    //1. Allocate unified memory
    struct container* dataUNI;
    cudaMallocManaged(&dataUNI, (sizeof(struct container) * size));

    //2. Transfer data from host to unified memory (Can do this with cpu operations)
    for (int i = 0; i < size; i++) {
        dataUNI[i] = data[i];
    }

    //3. Execute kernals "go my children, run with haste and do my bidding"
    printf("Sending out threads to do my dirty work\n");
    int blocks;
    if (size > 1024) {
        blocks = (size + (1024 - 1)) / 1024;
    }
    else {
        blocks = 1;
    }
    int device = NULL;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(dataUNI, (sizeof(struct container) * size), device);   //prefetch data for GPU
    tim.Start();
    operatorKernel <<<blocks, 1024>>> (dataUNI, size);
    tim.Stop();
    cudaDeviceSynchronize();
    float t_transfer = tim.Elapsed();

    // write results to output file
    char* newline = "\n";
    char buffer[10];
    char* res;
    FILE* solution = fopen(outputFile, "w");
    if (solution == NULL) {
        return -1;
    }
    for (int i = 0; i < size; i++) {
        res = itoa(dataUNI[i].result, buffer, 10);
        fprintf(solution, res);
        if (!(i == size - 1)) {
            fprintf(solution, newline);
        }
    }
    fclose(solution);
    cudaFree(dataUNI);

    printf("Kernel execution time: %f ms\n", t_transfer);
    return 0;
}
