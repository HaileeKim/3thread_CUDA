#include "./common.cpp"
#include <thread>
using std::thread;

// #include <nvtx3/nvtx3.hpp>

// input parameters
unsigned nrow = 10000; // num rows
unsigned ncol = 10000; // num columns

// host-side data
float* matA = nullptr;
float* matB = nullptr;
float* matC = nullptr;
float* matD = nullptr;
float* matE = nullptr;
float* matF = nullptr;

float** arrA = new float*[3];
float** arrB = new float*[3];
float** arrC = new float*[3];

int index_0;
int index_1;
int index_2;

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
		exit(EXIT_FAILURE); // ENOMEM: cannot allocate memory
	}
	// set random data
	srand( 0 );
	setNormalizedRandomData( arrA[index_0], nrow * ncol );
	setNormalizedRandomData( arrB[index_0], nrow * ncol );

	return NULL;
}

void *cudaFunction() 
{
	// kernel execution
	ELAPSED_TIME_BEGIN(0);
	for (register unsigned r = 0; r < nrow; ++r) {
		for (register unsigned c = 0; c < ncol; ++c) {
			unsigned i = r * ncol + c; // convert to 1D index
			arrC[index_1][i] = arrA[index_1][i] + arrB[index_1][i];
		}
	}
	ELAPSED_TIME_END(0);

	return NULL;
}

void *cpuFunction2() 
{

	// check the result
    printf("==cpuFunction2\n\n");
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
	printMat( "matC", arrC[index_2], nrow, ncol );
	printMat( "matA", arrA[index_2], nrow, ncol );
	printMat( "matB", arrB[index_2], nrow, ncol );
	// cleaning
	delete[] arrA[index_2];
	delete[] arrB[index_2];
	delete[] arrC[index_2];
	// done

	return NULL;
}


int main(const int argc, const char* argv[]) {
	

    cpuFunction1();
    cudaFunction();
    cpuFunction2();
    int i = 0;
	while(1) {
        index_0 = i % 3;
        index_1 = i % 3 + 1;
        index_2 = i % 3 + 2;

		thread t1(cpuFunction1);
		thread t2(cudaFunction);
        printf("====\n\n");
		thread t3(cpuFunction2);
        // printf("====\n\n");

		t1.join();
		t2.join();
		t3.join();

	}

	return 0;
}

