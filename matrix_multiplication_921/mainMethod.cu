
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "matrixMultiply.cu"
#include "matrixGenerator.cu"

#include <stdio.h>
#include <vector>

int main()
{
	
	// START - Randomized matrix creation
	srand(time(NULL));
    int multiple = rand() % 20; // a random multiple from 0 to 20
    printf("multiple = %d\n", multiple );
    
    const int width = 16 * multiple;
    printf("width = %d\n", width );
    
    int size = width * width;
    printf("size = %d\n", size );
    
	std::vector<float> M(size);
    for (int i = 0; i<size; i++) {
        M[i] = rand() * rand(); // a random value in the matrix from 0 to 1000
    }
	std::vector<float> N(size);
    for (int i = 0; i<size; i++) {
        N[i] = rand() * rand(); // a random value in the matrix from 0 to 1000
    }
	std::vector<float> P(size);
    // END - Randomized matrix creation
	


    
    const int arraySize = 32;
    float a[arraySize*arraySize];
	for(int i = 0; i < 1024; i++) {
		a[i] = i + 1;
	}
    float b[arraySize*arraySize];
	for(int i = 0; i < 1024; i++) {
		b[i] = i + 1;
	}
	float seq[arraySize*arraySize] = { 0 };
    float c[arraySize*arraySize] = { 0 };
	float d[arraySize*arraySize] = { 0 };
	float e[arraySize*arraySize] = { 0 };
	float f[arraySize*arraySize] = { 0 };
	float g[arraySize*arraySize] = { 0 };
	float h[arraySize*arraySize] = { 0 };
	float i[arraySize*arraySize] = { 0 };
	float timing2 = 0;
	float timing3 = 0;

	MatrixMulOnDevice1(a, b, seq, arraySize, NULL);
	MatrixMulOnDevice2(a, b, c, arraySize, timing2);
	MatrixMulOnDevice3(a, b, d, arraySize, timing3);
	MatrixMulOnDevice4(a, b, e, arraySize, 2);
	MatrixMulOnDevice4(a, b, f, arraySize, 4);
	MatrixMulOnDevice5(a, b, g, arraySize);
	MatrixMulOnDevice6(a, b, h, arraySize, 2);
	MatrixMulOnDevice6(a, b, i, arraySize, 4);

	printf("Sequential\n");
	printf("{%f\n,%f\n,%f\n,%f\n,%f\n,%f\n,%f\n,%f\n,%f\n}",
        seq[0], seq[1], seq[2], seq[3], seq[4], seq[5], seq[6], seq[7], seq[8], seq[9], seq[10], seq[11], seq[12], seq[13], seq[14], seq[15]);
	printf("\n");
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
	printf("\n");
	printf("{%f\n,%f\n,%f\n,%f\n,%f\n,%f\n,%f\n,%f\n,%f\n}",
        f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11], f[12], f[13], f[14], f[15]);
	printf("prefetch\n");
	printf("{%f\n,%f\n,%f\n,%f\n,%f\n,%f\n,%f\n,%f\n,%f\n}",
        g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8], g[9], g[10], g[11], g[12], g[13], g[14], g[15]);
	printf("Gran1x2\n");
	printf("{%f\n,%f\n,%f\n,%f\n,%f\n,%f\n,%f\n,%f\n,%f\n}",
        h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8], h[9], h[10], h[11], h[12], h[13], h[14], h[15]);	
	printf("Gran1x4\n");
	printf("{%f\n,%f\n,%f\n,%f\n,%f\n,%f\n,%f\n,%f\n,%f\n}",
        i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8], i[9], i[10], i[11], i[12], i[13], i[14], i[15]);
	getchar();

}
