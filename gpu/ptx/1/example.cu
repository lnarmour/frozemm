#include <stdio.h>
#include <stdlib.h>

__global__ void add(int *a) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
  a[index] += 1;
}

#define N (2048*2048)
#define MAX_INT 500
#define THREADS_PER_BLOCK 512

int main(void) {
	int *a;
	int *d_a;
	int size = sizeof(int) * N;

	cudaMalloc((void **)&d_a, size);

	// set up input valies
	a = (int *)malloc(size);
	for (int i=0; i<N; i++) {
		a[i] = 0;
	}

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	add<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a);
	cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);

	for (int i=0; i<N; i++) {
		printf("a[%d] = %d\n", i, a[i]);
		if (i == 20) break;
	}

	// cleanup
	free(a);
	cudaFree(d_a);

	return 0;
}

