
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t sumaCuda(int *d, const int *a, const int *b, const int *c);

//Kernel
__global__ void sumaKernel(int *d, const int *a, const int *b, const int *c)
{
    *d = *a + *b + *c;
}

int main()
{
	//Diego-Edgar Gracia Peña
	const int a = 4, b = 7, c = 3;
	int d = 0;

    // Add vectors in parallel.
    cudaError_t cudaStatus = sumaCuda(&d, &a, &b, &c);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Fallo en sumaCuda");
        return 1;
    }

    printf("%d + %d + %d = %d\n", a, b, c, d);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
	printf("Codificado por Diego-Edgar Gracia Peña");
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t sumaCuda(int *d, const int *a, const int *b, const int *c)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
	int *dev_d = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "¡fallo en cudaSetDevice!  ¿Seguro que tienes una GPU con capacidad para CUDA?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_d, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_c, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {//DEGP
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(dev_c, c, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

    // Launch a kernel on the GPU with one thread for each element.
    sumaKernel<<<1, 1>>>(dev_d, dev_a, dev_b, dev_c);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "fallo en el lanzamiento de sumaKernel: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(d, dev_d, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_d);
    cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
    
    return cudaStatus;
}
