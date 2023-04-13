#include "./common.cpp"
#include <thread>
using std::thread;

// input parameters
unsigned nrow = 10000; // num rows
unsigned ncol = 10000; // num columns

// host-side data
float* matA = nullptr;
float* matB = nullptr;
float* matC = nullptr;

// device-side data
float* dev_matA = nullptr;
float* dev_matB = nullptr;
float* dev_matC = nullptr;

float** arrA = new float*[3];
float** arrB = new float*[3];
float** arrC = new float*[3];

int index_0 = 0;
int index_1 = 1;
int index_2 = 2;

// CUDA kernel function
__global__ void kernel_matadd( float* c, const float* a, const float* b, unsigned nrow, unsigned ncol ) {
	unsigned col = blockIdx.x * blockDim.x + threadIdx.x; // CUDA-provided index
	unsigned row = blockIdx.y * blockDim.y + threadIdx.y; // CUDA-provided index
	if (row < nrow && col < ncol) {
		unsigned i = row * ncol + col; // converted to 1D index
		c[i] = a[i] + b[i];
	}
}

void *cpuFunction1() 
{
	try {
		matA = new float[nrow * ncol];
		matB = new float[nrow * ncol];
		matC = new float[nrow * ncol];

        arrA[index_0] = matA;
        arrB[index_0] = matB;
        arrC[index_0] = matC;

	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(EXIT_FAILURE);
	}
	// set random data
	srand( 0 );
	setNormalizedRandomData( arrA[index_0], nrow * ncol );
	setNormalizedRandomData( arrB[index_0], nrow * ncol );

	return NULL;
}

void *cudaFunction() 
{
	// allocate device memory
	ELAPSED_TIME_BEGIN(1);
	printf("\n\n=====\n\n");
	cudaMalloc( (void**)&dev_matA, nrow * ncol * sizeof(float) );
	printf("\n\n=====\n\n");
	cudaMalloc( (void**)&dev_matB, nrow * ncol * sizeof(float) );
	cudaMalloc( (void**)&dev_matC, nrow * ncol * sizeof(float) );
	CUDA_CHECK_ERROR();
	// copy to device from host
	cudaMemcpy( dev_matA, arrA[index_1], nrow * ncol * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_matB, arrB[index_1], nrow * ncol * sizeof(float), cudaMemcpyHostToDevice );
	CUDA_CHECK_ERROR();
	// CUDA kernel launch
	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid((ncol + dimBlock.x - 1) / dimBlock.x, (nrow + dimBlock.y - 1) / dimBlock.y, 1);
	CUDA_PRINT_CONFIG_2D( ncol, nrow );
	ELAPSED_TIME_BEGIN(0);
	kernel_matadd <<< dimGrid, dimBlock>>>( dev_matC, dev_matA, dev_matB, nrow, ncol );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);
	CUDA_CHECK_ERROR();
	// copy to host from device
	cudaMemcpy( arrC[index_1], dev_matC, nrow * ncol * sizeof(float), cudaMemcpyDeviceToHost );
	CUDA_CHECK_ERROR();
	ELAPSED_TIME_END(1);
	// free device memory
	cudaFree( dev_matA );
	cudaFree( dev_matB );
	cudaFree( dev_matC );
	CUDA_CHECK_ERROR();
	
	return NULL;
}

void *cpuFunction2() 
{

	// check the result
	float sumA = getSum( arrA[index_2], nrow * ncol );
	float sumB = getSum( arrB[index_2], nrow * ncol );
	float sumC = getSum( arrC[index_2], nrow * ncol );
	float diff = fabsf( sumC - (sumA + sumB) );
	printf("matrix size = nrow * ncol = %d * %d\n", nrow, ncol);
	printf("sumC = %f\n", sumC);
	printf("sumA = %f\n", sumA);
	printf("sumB = %f\n", sumB);
	printf("diff(sumC, sumA+sumB) =  %f\n", diff);
	printf("diff(sumC, sumA+sumB) / (nrow * ncol) =  %f\n", diff / (nrow * ncol));
	printf("arrC[%d] : ", index_2);
	printMat( "arrC", arrC[index_2], nrow, ncol );
	printf("arrA[%d] : ", index_2);
	printMat( "matA", arrA[index_2], nrow, ncol );
	printf("arrB[%d] : ", index_2);
	printMat( "matB", arrB[index_2], nrow, ncol );
	// cleaning
	delete[] arrA[index_2];
	delete[] arrB[index_2];
	delete[] arrC[index_2];

	return NULL;
}


int main(const int argc, const char* argv[]) {

	index_0 = 0;
    cpuFunction1();
	
	index_0 = 1;
	index_1 = 0;
    cpuFunction1();
    cudaFunction();
    
	index_0 = 2;
	index_1 = 1;
	index_2 = 0;
    cpuFunction1();
    cudaFunction();
	cpuFunction2();
	
	int i = 0;
	
	while(1) 
	{
		printf("\n ----start Cycle---- \n");

        index_0 = (i) % 3;
        index_1 = (i + 1) % 3;
        index_2 = (i + 2) % 3;

		thread t1(cpuFunction1);
		thread t2(cudaFunction);
		thread t3(cpuFunction2);

		t1.join();
		t2.join();
		t3.join();

		i++;
	}

	// done
	return 0;
}

