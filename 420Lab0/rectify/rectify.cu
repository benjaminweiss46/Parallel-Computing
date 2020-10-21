#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "lodepng.h"//including the given c file for image data extraction
#include <time.h>


__global__ void rectifyKernel(unsigned char* input, unsigned width, unsigned height, int threads) {
    //number of pixels each thread will do initially
    int taskSize = ((width * height * 4) / threads);
    //need to adjust for dimensions of our grid
    int adj = (threadIdx.x + (blockDim.x*blockIdx.x)) * taskSize;
    /**
    * Have each thread do (width*height*4)/threads) but if (width*height*4)%threads isn't 0 there will be some left over threads 
    * will deal with this later
    */
    unsigned char barrier = 127;
    for (int i = 0; i < taskSize; i++) {
        int adj_i = adj + i;
        if (input[adj_i] < barrier) {
          input[adj_i] = barrier;
        }
    }
    //Now some threads will need to do an extra one
    int leftOver = ((width * height * 4) % threads);
    if ((leftOver != 0) && ((threadIdx.x + (blockDim.x * blockIdx.x)) <= leftOver)) {
        if (input[taskSize * threads + threadIdx.x] < 127) {
            input[taskSize * threads + threadIdx.x] = 127;
        }
    }
    
}

void rectify(unsigned char* input,char* output, unsigned width, unsigned height, int threads) {
    //Note: looked up the max number of threads per block, 1024
    int blocks;
    if (threads > 1024) {
        blocks = (threads + (1024 - 1)) / 1024;
    }
    else {
        blocks = 1;
    }
    printf("Will rectify using %d threads contained in %d blocks\n", threads, blocks);

    //From Tutorial
    //1. allocate host memory and initialize host data
    //unsigned char* input is our host data
    //but will create a new image memory space for copying back later
    unsigned char* copyInput;
    copyInput = (unsigned char*)malloc(width*height*4*sizeof(unsigned char));
    printf("First 5 pixels from CPU mem is,\n");
    for (int i = 0; i < 5; i++) {
        printf("%u, ", input[i]);
    }
    printf("\n");
    //2. Allocate device memory
    unsigned char* inputGPU;
    cudaMalloc((void**)&inputGPU, (width*height*4*sizeof(unsigned char)));

    //3. Transfer data from host to device memory
    cudaMemcpy(inputGPU, input, (width*height*4*sizeof(unsigned char)), cudaMemcpyHostToDevice);

    //4. Execute kernals "go my children, run with haste and do my bidding"
    printf("Sending out threads to do my dirty work\n");
    //From tutorial (timing)
    float memsettime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    rectifyKernel<<<blocks, (threads / blocks)>>> (inputGPU, width, height, threads);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&memsettime, start, stop);
    printf("*** CUDA execution time: %f *** \n", memsettime);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    //5. transfer output from device memory to host memory 
    cudaMemcpy(copyInput, inputGPU, (width*height*4*sizeof(unsigned char)), cudaMemcpyDeviceToHost);
    printf("First 5 pixels from CPU mem (UPDATED) is,\n");
    for (int i = 0; i < 5; i++) {
        printf("%u, ", copyInput[i]);
    }
    printf("\n");
    printf("Creating output image\n");
    lodepng_encode32_file(output, copyInput, width, height);
    printf("Thanks for rectifying, come again\n");
    free(input);
    free(copyInput);
    cudaFree(inputGPU);
}

int main(int argc, char* argv[])
{   
    //declare all vars used in main
    unsigned error;
    unsigned char* image;
    unsigned width, height;

    //Take argument inputs
    char* input_filename = argv[1];
    char* output_filename = argv[2];
    char* numThreadsInput = argv[3];
    int numThreads = atoi(numThreadsInput); //prefer to work with int in this case, use when taking command line arguments
    if (numThreads < 1) {
        fprintf(stderr, "Number of threads argument must be a valid integer.");
        return 1;
    }
    //load the image
    error = lodepng_decode32_file(&image, &width, &height, input_filename);
    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
    else printf("Image loaded successfully!\n");
    if (numThreads == 1) {
         printf("Taking %s running rectification with %d threads and returning new file %s\n", input_filename, numThreads, output_filename);
         clock_t start = clock();
         for (int i = 0; i < (width*height*4); i++) {
             if (image[i] < 127) {
                image[i] = 127;
             }
         }
         clock_t time = clock() - start;
         printf("Rectification with 1 thread took %d long\n", time);
     }
    else {
         printf("Taking %s running rectification with %d threads and returning new file %s\n", input_filename, numThreads, output_filename);
         //Lets start rectifying
         rectify(image, output_filename, width, height, numThreads);
     }



    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}


