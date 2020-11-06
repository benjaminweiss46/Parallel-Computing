#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "lodepng.h"//including the given c file for image data extraction
#include <time.h>
#include "wm.h"
#include <math.h>


__global__ void convolveKernel(unsigned char* inputReference, unsigned char* inputChange, unsigned width, unsigned height, int threads, float * w) {
    int outPutHeight = (height - 2);
    int outPutWidth = (width - 2);
    /*
     *number of pixels each thread will need to do initially in the changed picture
     * Ignore RGB for now will mean we each thread will deal with RGB for all pixels it is resposible for this
     * is a slower paralization scheme only in the case where threads > ((outPutHeight * outPutWidth) / threads) 
     * also leads to undefined behaviour. I think this is a fine caveat in saving me some time, the assumption
     * that threads won't be greater 9,203,524 is fair for this course. 
     */
    int taskSize = ((outPutHeight * outPutWidth) / threads); //num output pixels / thread
    // Each thread will do portion of work from (thread global location * tasksize)->(thread global location * tasksize) + tasksize
    int startPix = (threadIdx.x + (blockDim.x * blockIdx.x)) * taskSize; //starting location in output image
    /*
    * Have each thread do (outputwidth*outputheight*3)/threads) but if (outputwidth*outputheight*3)%threads isn't 0 there will be some left over threads
    * will deal with this later.
    * 
    * If I want to do convolution on pixel 0 of inputChange, I need data from pixels 0,1,2,3840,3841,3842,7680,7681,7681 in inputRef
    * multiply by 4 to get location in list and add 0,1,2 for each different value R,G,B
    * since our images are different sizes we must consider the conversion, all pixels shifted 1 down, 1 right
    * pixel 0 in output = pixel 3841 in original
    *       9000        =       12841
    *       opx         =   inpx + 3840 + 1
    * this brings us to our next equation for input pixel i we need pixel i+-3840, i+-1, i +- 3840 +-1 and i itself
    * hope I explained it right, its a bit confusing
    */
    int outPix; //Current Pixel (output)
    int outLoc; //Current location of pixel in list (output)
    int outRow; 
    int outCol;
    int inPix; //Current Pixel (input, larger)
    int inLoc; //Current location of pixel in list (input, larger image)
    int type; // R = 0, G = 1, B = 1
    float x;
    for (int a = 0; a < 3; a++) {
        for (int i = 0; i < taskSize; i++) {
            outPix = startPix + i; //Get current pixel in output image
            outCol = outPix % outPutWidth; //Get current pixel col in output image
            outRow = (outPix - outCol) / outPutWidth; //Get current pixel row in output image
            inPix = ((outRow + 1) * width) + outCol + 1; //Get current pixel in input image
            outLoc = outPix * 4 + a; //convert both: Pixel->Item in PNG List
            inLoc = inPix * 4 + a;
            if (a == 2) {
                inputChange[outLoc+1] = inputReference[inLoc+1]; //just set the alpha value to what it was before when you are working on G
            }
            x = inputReference[inLoc - 4 - width*4] * w[0] + inputReference[inLoc - width * 4] * w[1] + inputReference[inLoc + 4 - width * 4] * w[2] + //Top row
                inputReference[inLoc - 4] * w[3] + inputReference[inLoc] * w[4] + inputReference[inLoc + 4] * w[5] + //mid row
                inputReference[inLoc - 4 + width * 4] * w[6] + inputReference[inLoc + width * 4] * w[7] + inputReference[inLoc + 4 + width * 4] * w[8]; //bot row
            if (x < 0) {
                x = 0;
            }
            else if (x > 255) {
                x = 255;
            }
            
            inputChange[outLoc] = (unsigned char) ((int) round(x));
        }
    }
    //now some threads will need to do an extra one, in some cases 
    int leftover = ((outPutHeight * outPutWidth) % threads);
    if ((leftover != 0) && ((threadIdx.x + (blockDim.x * blockIdx.x)) <= leftover)) {
        for (int a = 0; a < 3; a++) {
                outPix = taskSize * threads + threadIdx.x; //get current pixel in output image
                outCol = outPix % outPutWidth; //get current pixel col in output image
                outRow = (outPix - outCol) / outPutWidth; //get current pixel row in output image
                inPix = ((outRow + 1) * width) + outCol + 1; //get current pixel in input image
                outLoc = outPix * 4 + a; //convert both: pixel->item in png list
                inLoc = inPix * 4 + a;
                if (a == 2) {
                    inputChange[outLoc + 1] = inputReference[inLoc + 1]; //just set the alpha value to what it was before when you are working on G
                }
                x = inputReference[inLoc - 4 - width * 4] * w[0] + inputReference[inLoc - width * 4] * w[1] + inputReference[inLoc + 4 - width * 4] * w[2] + //Top row
                    inputReference[inLoc - 4] * w[3] + inputReference[inLoc] * w[4] + inputReference[inLoc + 4] * w[5] + //mid row
                    inputReference[inLoc - 4 + width * 4] * w[6] + inputReference[inLoc + width * 4] * w[7] + inputReference[inLoc + 4 + width * 4] * w[8]; //bot row
                if (x < 0) {
                    x = 0;
                }
                else if (x > 255) {
                    x = 255;
                }
                inputChange[outLoc] = (unsigned char)((int)round(x));
        }
    }

}

void convolve(unsigned char* input, char* output, unsigned width, unsigned height, int threads) {
    //Note: looked up the max number of threads per block, 1024
    int blocks;
    if (threads > 1024) {
        blocks = (threads + (1024 - 1)) / 1024;
    }
    else {
        blocks = 1;
    }
    printf("Will convolve using %d threads contained in %d blocks\n", threads, blocks);
    //From Tutorial
    //1. allocate unified memory and need two areas, one for image input and one for image output since for convolution we can't edit image in place
    printf("\nAllocating unified memory..\n");
    unsigned char* imageDataUNI; //Data for referencing
    cudaMallocManaged(&imageDataUNI, (width * height * 4 * sizeof(unsigned char)));


    unsigned char* imageChangeDataUNI; //data for changing, smaller since we don't do sides
    cudaMallocManaged(&imageChangeDataUNI, ((width - 2) * (height - 2) * 4 * sizeof(unsigned char)));


    float * weights; //load weights into unified memory
    cudaMallocManaged(&weights, (9 * sizeof(float)));

    //2. Transfer data from host to unified memory (Can do this with cpu operations)
    printf("\nTranfering data for reference to unified memory..\n");
    for (int i = 0; i < (width * height * 4 * sizeof(unsigned char)); i++) {
        imageDataUNI[i] = input[i];
    }
    for (int i = 0; i < ((width - 2) * (height - 2) * 4 * sizeof(unsigned char)); i++) {
        imageChangeDataUNI[i] = (unsigned char) 0;
    }
    int count = 0;
    for (int a = 0; a<3; a++) {
        for (int b = 0; b < 3; b++) {
            weights[count] = w[a][b];
            count++;
        }
    }

    printf("\nConfirm correct transfer..\n");
    printf("First 5 pixels from CPU mem is,\n");
    for (int i = 0; i < 5; i++) {
        printf("%u, ", input[i]);
    }
    printf("\n");

    printf("First 5 pixels from UNIfied reference mem is,\n");
    for (int i = 0; i < 5; i++) {
        printf("%u, ", imageDataUNI[i]);
    }
    printf("\n");

    printf("First 5 pixels from UNIfied change mem is,\n");
    for (int i = 0; i < 5; i++) {
        printf("%u, ", imageChangeDataUNI[i]);
    }
    printf("\n");

    printf("Weights list from unified memory is,\n");
    for (int i = 0; i < 9; i++) {
        printf("%f, ", weights[i]);
    }
    printf("\n");

    //3. Execute kernals "go my children, run with haste and do my bidding"
    printf("Sending out threads to do my dirty work..\n");
    convolveKernel << <blocks, (threads / blocks) >> > (imageDataUNI, imageChangeDataUNI, width, height, threads, weights);
    cudaDeviceSynchronize();
    printf(cudaGetErrorString(cudaGetLastError()));

    printf("\nCreating output image\n");
    lodepng_encode32_file(output, imageChangeDataUNI, width-2, height-2);
    printf("Thanks for convolving, come again\n");

    free(input);
    cudaFree(imageChangeDataUNI);
    cudaFree(imageDataUNI);
    cudaFree(weights);
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

    printf("Taking %s running convolution with %d threads and returning new file %s\n", input_filename, numThreads, output_filename);
    printf("Height: %d, Width: %d\n", height, width);
    //Lets start convolution
    convolve(image, output_filename, width, height, numThreads);



    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}


