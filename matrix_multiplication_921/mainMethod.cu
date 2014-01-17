
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "matrixMultiply.cu"
#include "matrixGenerator.cu"

#include <stdio.h>

int main()
{

	// START - Randomized matrix creation
	srand(time(NULL));
    int multiple = rand() % 20; // a random multiple from 0 to 20
    printf("multiple = %d\n", multiple );
    
    int width = 16 * multiple;
    printf("width = %d\n", width );
    
    int size = width * width;
    printf("size = %d\n", size );
    
    int M[size];
    for (int i = 0; i<size; i++) {
        M[i] = random() % 1000; // a random value in the matrix from 0 to 1000
    }
    int N[size];
    for (int i = 0; i<size; i++) {
        N[i] = random() % 1000; // a random value in the matrix from 0 to 1000
    }
    int P[size];
    // END - Randomized matrix creation


    
    const int arraySize = 512;
    float a[arraySize*arraySize];
	for(int i = 0; i < 262144; i++) {
		a[i] = 2;
	}
    float b[arraySize*arraySize];
	for(int i = 0; i < 262144; i++) {
		b[i] = 2;
	}
    float c[arraySize*arraySize] = { 0 };
	float d[arraySize*arraySize] = { 0 };
	float e[arraySize*arraySize] = { 0 };
	float f[arraySize*arraySize] = { 0 };
	float timing2 = 0;
	float timing3 = 0;

	MatrixMulOnDevice2(a, b, c, arraySize, timing2);
	MatrixMulOnDevice3(a, b, d, arraySize, timing3);
	MatrixMulOnDevice4(a, b, e, arraySize, 2);
	MatrixMulOnDevice4(a, b, f, arraySize, 4);

	printf("{%f\n,%f\n,%f\n,%f\n,%f\n,%f\n,%f\n,%f\n,%f\n}",
        c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11], c[12], c[13], c[14], c[15]);
	printf("The time is: %f", timing2);
	printf("\n");
	printf("{%f\n,%f\n,%f\n,%f\n,%f\n,%f\n,%f\n,%f\n,%f\n}",
        d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15]);
	printf("The time is: %f", timing3);
	printf("\n");
	printf("{%f\n,%f\n,%f\n,%f\n,%f\n,%f\n,%f\n,%f\n,%f\n}",
        e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9], e[10], e[11], e[12], e[13], e[14], e[15]);
	printf("{%f\n,%f\n,%f\n,%f\n,%f\n,%f\n,%f\n,%f\n,%f\n}",
        f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11], f[12], f[13], f[14], f[15]);
	getchar();

}
