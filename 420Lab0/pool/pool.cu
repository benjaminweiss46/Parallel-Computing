#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "lodepng.h"


// typedefs
typedef unsigned int uint;
typedef unsigned char uchar;

// constants
constexpr auto DEVICE = 0;
constexpr uint PIXEL_SIZE = 4 * sizeof(uchar);


__global__ void pool(uchar* image, uchar* new_image, uint width_px, uint size_pools, uint n_threads) {
	// calculate pools per thread and width in pools
	uint pools_per_thread = size_pools / n_threads;
	uint width_pools = width_px / 2;

	// set first pool index and pool index limit
	uint pool = threadIdx.x * pools_per_thread;
	uint pool_lim = pool + pools_per_thread;
	if (pool_lim >= size_pools || threadIdx.x == n_threads - 1) {
		pool_lim = size_pools;
	}
	
	while (pool < pool_lim) {
		// get top-left px index of pool
		uint x = pool % width_pools;
		uint y = pool / width_pools;
		uint px = 2 * (x + y * width_px);
		// note: px index for new image is equivalent to pool index

		// get indices of first uchar for each px in the pool
		uint pool_ind[] = { px * PIXEL_SIZE, (px + 1) * PIXEL_SIZE, (px + width_px) * PIXEL_SIZE, (px + width_px + 1) * PIXEL_SIZE };

		// perform max pooling
		for (int c = 0; c < 4; c++) {
			uchar pool_max = image[pool_ind[0] + c];
			for (uint ind_0 : pool_ind) {
				uchar val = image[ind_0 + c];
				if (val > pool_max) {
					pool_max = val;
				}
			}
			new_image[pool * PIXEL_SIZE + c] = pool_max;
		}

		// increment pool index
		pool++;
	}	
}

int main(int argc, char* argv[]) {
	clock_t runtime_start = clock();

	// parse command line arguments
	if (argc != 4) {
		fprintf(stderr, "Incorrect number of arguments. Correct sytnax: <input filename> <output filename> <number of threads>");
		return 1;
	}
	const char *input_filename = argv[1];
	const char *output_filename = argv[2];
	int n_threads = 0;
	try {
		n_threads = std::stoi(argv[3]);
	}
	catch (const std::exception&) {
		fprintf(stderr, "Number of threads argument must be a valid integer.");
		return 1;
	}

	// load png file
	uint error;
	uchar *image;
	uint width, height;

	error = lodepng_decode32_file(&image, &width, &height, input_filename);
	if (error) {
		fprintf(stderr, "error %u: %s\n", error, lodepng_error_text(error));
		return 1;
	}

	// set device
	cudaError_t cudaStatus = cudaSetDevice(DEVICE);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed. Make sure that a CUDA-capable GPU is installed.");
		return 1;
	}

	//get device properties
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, DEVICE);

	// confirm that number of threads does not exceed maximum
	if (n_threads > props.maxThreadsDim[0]) {
		fprintf(stderr, "This program cannot run over %i threads.", props.maxThreadsDim[0]);
		return 1;
	}

	// set up unified memory for input and output images, thread distribution
	uchar *uf_image, *uf_new_image;
	uint image_size = width * height * PIXEL_SIZE;
	uint new_image_size = width / 2 * height / 2 * PIXEL_SIZE;

	cudaStatus = cudaMallocManaged(&uf_image, image_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMallocManaged failed with error %d (%s).", cudaStatus, cudaGetErrorString(cudaStatus));
		free(image);
		return 1;
	}
	memcpy(uf_image, image, image_size);

	cudaStatus = cudaMallocManaged(&uf_new_image, new_image_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMallocManaged failed with error %d (%s).", cudaStatus, cudaGetErrorString(cudaStatus));
		free(image);
		return 1;
	}

	// get size of image in 2x2 groups of pixels
	// this division assumes both dimensions are divisible by 2
	uint size_pools = width * height / 4;

	// launch kernel
	clock_t gpu_start = clock();
	pool<<< 1, n_threads >>>(uf_image, uf_new_image, width, size_pools, n_threads);

	// check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		free(image);
		cudaFree(uf_image);
		cudaFree(uf_new_image);
		return 1;
	}

	// wait for GPU threads to complete
	cudaDeviceSynchronize();

	float tim_gpu = (float) (clock() - gpu_start) / CLOCKS_PER_SEC;

	// check for errors that occured during the launch.
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel.\n", cudaStatus);
		free(image);
		cudaFree(uf_image);
		cudaFree(uf_new_image);
		return 1;
	}

	// encode and create file
	lodepng_encode32_file(output_filename, uf_new_image, width / 2, height / 2);

	free(image);
	cudaFree(uf_image);
	cudaFree(uf_new_image);

	float tim_runtime = (float)(clock() - runtime_start) / CLOCKS_PER_SEC;

	printf("Kernel runtime: %f s\n", tim_gpu);
	printf("Total runtime: %f s\n", tim_runtime);

	return 0;
}