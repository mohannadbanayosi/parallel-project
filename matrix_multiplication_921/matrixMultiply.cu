#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

#include <stdio.h>
#include <assert.h>

#ifndef __CUDACC__
#define __CUDACC__
#endif

inline __global__ void MatrixMulKernelTiled8x8(float* Md, float* Nd, float* Pd, int Width);
inline __global__ void MatrixMulKernelTiled16x16(float* Md, float* Nd, float* Pd, int Width);
inline __global__ void MatrixMulKernelTiledUnrolling(float* Md, float* Nd, float* Pd, int Width, int unrolling);



inline void MatrixMulOnDevice2(float* M, float* N, float* P, int Width, float timing) {   
	int size = Width * Width * sizeof(float);    
	float *Md, *Nd, *Pd;   
	//Allocate and Load M, N to device memory    
	cudaMalloc(&Md, size);   
	cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);   
	cudaMalloc(&Nd, size);   
	cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);   
	//Allocate P on the device   
	cudaMalloc(&Pd, size);
	// Kernel invocation code – to be shown later    
	// Setup the execution configuration    
	dim3 dimGrid2(Width/8, Width/8);   
	dim3 dimBlock2(8, 8);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start record
	cudaEventRecord(start, 0);
	// Do something on GPU
    // Launch the device computation threads!    
	MatrixMulKernelTiled8x8<<< dimGrid2, dimBlock2 >>>(Md, Nd, Pd, Width);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timing, start, stop); // that's our time!
	// Clean up:
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	//Read P from the device    
	cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);        
	// Free device matrices    
	cudaFree(Md); cudaFree(Nd); cudaFree (Pd); 
}

inline void MatrixMulOnDevice3(float* M, float* N, float* P, int Width, float timing) {   
	int size = Width * Width * sizeof(float);    
	float *Md, *Nd, *Pd;   
	//Allocate and Load M, N to device memory    
	cudaMalloc(&Md, size);   
	cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);   
	cudaMalloc(&Nd, size);   
	cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);   
	//Allocate P on the device   
	cudaMalloc(&Pd, size);
	// Kernel invocation code – to be shown later    
	// Setup the execution configuration    
	dim3 dimGrid3(Width/16, Width/16);   
	dim3 dimBlock3(16, 16);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start record
	cudaEventRecord(start, 0);
	// Do something on GPU
    // Launch the device computation threads!  
    // Launch the device computation threads!    
	MatrixMulKernelTiled16x16<<< dimGrid3, dimBlock3 >>>(Md, Nd, Pd, Width);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timing, start, stop); // that's our time!
	// Clean up:
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	//Read P from the device    
	cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);        
	// Free device matrices    
	cudaFree(Md); cudaFree(Nd); cudaFree (Pd); 
}

inline void MatrixMulOnDevice4(float* M, float* N, float* P, int Width, int unrolling) {   
	int size = Width * Width * sizeof(float);    
	float *Md, *Nd, *Pd;   
	//Allocate and Load M, N to device memory    
	cudaMalloc(&Md, size);   
	cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);   
	cudaMalloc(&Nd, size);   
	cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);   
	//Allocate P on the device   
	cudaMalloc(&Pd, size);
	// Kernel invocation code – to be shown later    
	// Setup the execution configuration    
	dim3 dimGrid3(Width/8, Width/8);   
	dim3 dimBlock3(8, 8);

    // Launch the device computation threads!    
	MatrixMulKernelTiledUnrolling<<< dimGrid3, dimBlock3 >>>(Md, Nd, Pd, Width, unrolling);

	//Read P from the device    
	cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);        
	// Free device matrices    
	cudaFree(Md); cudaFree(Nd); cudaFree (Pd); 
}

// Matrix multiplication kernel – per thread code
/*inline __global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int Width) {        
	// Pvalue is used to store the element of the matrix    
	// that is computed by the thread    
	float Pvalue = 0;
	for (int k = 0; k < Width; ++k) {    
		float Melement=Md[threadIdx.y*Width+k];    
		float Nelement=Nd[k*Width+threadIdx.x];    
		Pvalue += Melement * Nelement;  
	}  
	Pd[threadIdx.y*Width+threadIdx.x]=Pvalue; 
}*/


inline __global__ void MatrixMulKernelTiled8x8(float* Md, float* Nd, float* Pd, int Width) { 
	const int TILE_WIDTH = 8;
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH]; 
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH]; 
	int bx = blockIdx.x;  
	int by = blockIdx.y; 
	int tx = threadIdx.x; 
	int ty = threadIdx.y; 
	// Identify the row and column of the Pd element to work on
	int Row = by * TILE_WIDTH + ty; 
	int Col = bx * TILE_WIDTH + tx; 
	float Pvalue = 0; 
	// Loop over the Md and Nd tiles required to compute the Pd element 
	for (int m = 0; m < Width/TILE_WIDTH; ++m) { 
		// Collaborative loading of Md and Nd tiles into shared memory
		Mds[ty][tx] = Md[Row*Width + (m*TILE_WIDTH + tx)]; 
		Nds[ty][tx] = Nd[(m*TILE_WIDTH + ty)*Width + Col];
		__syncthreads();
		for (int k = 0; k < TILE_WIDTH; ++k) 
			Pvalue += Mds[ty][k] * Nds[k][tx];
		__syncthreads();
	} 
	Pd[Row*Width + Col] = Pvalue; 
}

inline __global__ void MatrixMulKernelTiled16x16(float* Md, float* Nd, float* Pd, int Width) { 
	const int TILE_WIDTH = 16;
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH]; 
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH]; 
	int bx = blockIdx.x;  
	int by = blockIdx.y; 
	int tx = threadIdx.x; 
	int ty = threadIdx.y; 
	// Identify the row and column of the Pd element to work on
	int Row = by * TILE_WIDTH + ty; 
	int Col = bx * TILE_WIDTH + tx; 
	float Pvalue = 0; 
	// Loop over the Md and Nd tiles required to compute the Pd element 
	for (int m = 0; m < Width/TILE_WIDTH; ++m) { 
		// Collaborative loading of Md and Nd tiles into shared memory
		Mds[ty][tx] = Md[Row*Width + (m*TILE_WIDTH + tx)]; 
		Nds[ty][tx] = Nd[(m*TILE_WIDTH + ty)*Width + Col];
		__syncthreads();
		for (int k = 0; k < TILE_WIDTH; ++k) 
			Pvalue += Mds[ty][k] * Nds[k][tx];
		__syncthreads();
	} 
	Pd[Row*Width + Col] = Pvalue; 
}

inline __global__ void MatrixMulKernelTiled8x8Prefetching(float* Md, float* Nd, float* Pd, int Width) { 

}

//inline __global__ void MatrixMulKernelTiledUnrolling(float* Md, float* Nd, float* Pd, int Width, int unrolling) { 
//	const int TILE_WIDTH = 8;
//	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH]; 
//	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH]; 
//	int bx = blockIdx.x;  
//	int by = blockIdx.y; 
//	int tx = threadIdx.x; 
//	int ty = threadIdx.y; 
//	// Identify the row and column of the Pd element to work on
//	int Row = by * TILE_WIDTH + ty; 
//	int Col = bx * TILE_WIDTH + tx; 
//	float Pvalue = 0; 
//	// Loop over the Md and Nd tiles required to compute the Pd element 
//	for (int m = 0; m < Width/TILE_WIDTH; ++m) { 
//		// Collaborative loading of Md and Nd tiles into shared memory
//		Mds[ty][tx] = Md[Row*Width + (m*TILE_WIDTH + tx)]; 
//		Nds[ty][tx] = Nd[(m*TILE_WIDTH + ty)*Width + Col];
//		__syncthreads();
//		for (int k = 0; k < TILE_WIDTH; k+=unrolling) {
//			if(unrolling == 2) {
//				Pvalue += Mds[ty][k] * Nds[k][tx];
//				Pvalue += Mds[ty][k + 1] * Nds[k + 1][tx];
//			}
//			else if(unrolling == 4) {
//				Pvalue += Mds[ty][k] * Nds[k][tx];
//				Pvalue += Mds[ty][k + 1] * Nds[k + 1][tx];
//				Pvalue += Mds[ty][k + 2] * Nds[k + 2][tx];
//				Pvalue += Mds[ty][k + 3] * Nds[k + 3][tx];
//			}
//			else {
//			}
//		}
//		__syncthreads();
//	} 
//	Pd[Row*Width + Col] = Pvalue; 
//}

inline __global__ void MatrixMulKernelTiledUnrolling(float* Md, float* Nd, float* Pd, int Width, int unrolling) { 
	const int TILE_WIDTH = 8;
	int bx = blockIdx.x;  
	int by = blockIdx.y; 
	int tx = threadIdx.x; 
	int ty = threadIdx.y; 
	// Identify the row and column of the Pd element to work on
	int Row = by * TILE_WIDTH + ty; 
	int Col = bx * TILE_WIDTH + tx; 
	float Pvalue = 0; 
	// Loop over the Md and Nd tiles required to compute the Pd element 
	 for (int k = 0; k < Width; ++k) {    
		 float Melement=Md[threadIdx.y*Width+k];    
		 float Nelement=Nd[k*Width+threadIdx.x];    
		 Pvalue += Melement * Nelement;  
	 }

	Pd[Row*Width + Col] = Pvalue; 


inline __global__ void MatrixMulKernelTiled8x8gran1x2(float* Md, float* Nd, float* Pd, int Width) { 
	const int TILE_WIDTH = 8;
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH]; 
	__shared__ float Nds1[TILE_WIDTH][TILE_WIDTH]; 
	__shared__ float Nds2[TILE_WIDTH][TILE_WIDTH]; 
	int bx = blockIdx.x;  
	int by = blockIdx.y; 
	int tx = threadIdx.x; 
	int ty = threadIdx.y; 
	// Identify the row and column of the Pd element to work on
	int Row = by * TILE_WIDTH + ty; 
	int Col1 = bx * TILE_WIDTH + tx; 
	int Col2 = Col1 + TILE_WIDTH; 
	float Pvalue1 = 0; 
	float Pvalue2 = 0; 
	// Loop over the Md and Nd tiles required to compute the Pd element 
	for (int m = 0; m < Width/TILE_WIDTH; ++m) { 
		// Collaborative loading of Md and Nd tiles into shared memory
		Mds[ty][tx] = Md[Row*Width + (m*TILE_WIDTH + tx)]; 
		Nds1[ty][tx] = Nd[(m*TILE_WIDTH + ty)*Width + Col1];
		Nds2[ty][tx] = Nd[(m*TILE_WIDTH + ty)*Width + Col2];
		__syncthreads();
		for (int k = 0; k < TILE_WIDTH; ++k) 
			Pvalue1 += Mds[ty][k] * Nds1[k][tx];
			Pvalue2 += Mds[ty][k] * Nds2[k][tx];
		__syncthreads();
	} 
	Pd[Row*Width + Col1] = Pvalue1; 
	Pd[Row*Width + Col2] = Pvalue2; 
}

inline __global__ void MatrixMulKernelTiled8x8gran1x4(float* Md, float* Nd, float* Pd, int Width) { 
	const int TILE_WIDTH = 8;
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH]; 
	__shared__ float Nds1[TILE_WIDTH][TILE_WIDTH]; 
	__shared__ float Nds2[TILE_WIDTH][TILE_WIDTH]; 
	__shared__ float Nds3[TILE_WIDTH][TILE_WIDTH]; 
	__shared__ float Nds4[TILE_WIDTH][TILE_WIDTH]; 
	int bx = blockIdx.x;  
	int by = blockIdx.y; 
	int tx = threadIdx.x; 
	int ty = threadIdx.y; 
	// Identify the row and column of the Pd element to work on
	int Row = by * TILE_WIDTH + ty; 
	int Col1 = bx * TILE_WIDTH + tx; 
	int Col2 = Col1 + TILE_WIDTH; 
	int Col3 = Col2 + TILE_WIDTH; 
	int Col4 = Col3 + TILE_WIDTH; 
	float Pvalue1 = 0; 
	float Pvalue2 = 0; 
	float Pvalue3 = 0; 
	float Pvalue4 = 0; 
	// Loop over the Md and Nd tiles required to compute the Pd element 
	for (int m = 0; m < Width/TILE_WIDTH; ++m) { 
		// Collaborative loading of Md and Nd tiles into shared memory
		Mds[ty][tx] = Md[Row*Width + (m*TILE_WIDTH + tx)]; 
		Nds1[ty][tx] = Nd[(m*TILE_WIDTH + ty)*Width + Col1];
		Nds2[ty][tx] = Nd[(m*TILE_WIDTH + ty)*Width + Col2];
		Nds3[ty][tx] = Nd[(m*TILE_WIDTH + ty)*Width + Col3];
		Nds4[ty][tx] = Nd[(m*TILE_WIDTH + ty)*Width + Col4];
		__syncthreads();
		for (int k = 0; k < TILE_WIDTH; ++k) 
			Pvalue1 += Mds[ty][k] * Nds1[k][tx];
			Pvalue2 += Mds[ty][k] * Nds2[k][tx];
			Pvalue3 += Mds[ty][k] * Nds3[k][tx];
			Pvalue4 += Mds[ty][k] * Nds4[k][tx];
		__syncthreads();
	} 
	Pd[Row*Width + Col1] = Pvalue1; 
	Pd[Row*Width + Col2] = Pvalue2; 
	Pd[Row*Width + Col3] = Pvalue3; 
	Pd[Row*Width + Col4] = Pvalue4; 
}