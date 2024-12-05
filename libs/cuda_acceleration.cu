#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#include "cuda_acceleration.h"

#define BLOCKS		1024
#define THREADS		1024

// compile with  /usr/local/cuda/bin/nvcc /home/user/tiff_file_parser/libs/cuda_acceleration.cu -o /home/user/tiff_file_parser/cuda_acceleration.o -c

__device__ static inline signed long limit(signed long input, signed long lower, signed long upper);
__global__ void ParalellTotalContrastParallelismFunction (float **InputData, float **outputData, unsigned int frame, unsigned long index, unsigned int hResolution, unsigned int vResolution);
float cppParalellTotalContrast(float **image, float radius, unsigned int vResolution, unsigned int hResolution);
void printStatusBar(unsigned char input)
{
	printf("[");
	for (unsigned int i = 0; i < 50; i++)
	{
		if (i > input / 2)
		{
			printf(" ");
		}
		else
		{
			printf("#");
		}
	}
	printf("]");
	fflush(stdout);
}

extern "C"{
float ParalellTotalContrast(float **image, float radius, unsigned int vResolution, unsigned int hResolution)
{
	return cppParalellTotalContrast(image, radius, vResolution, hResolution);
}
}

float cppParalellTotalContrast(float **image, float radius, unsigned int vResolution, unsigned int hResolution)
{
	unsigned int frame = radius * hResolution;
	float returnval = 0;
	float ** cu_image;
	float ** output;
	unsigned long index = 0;

	printf("calculating avg contrast: \n");
	printf("trying to allocate %i bytes of type float with sizeof %li resulting in %li bytes allocated \n", hResolution, sizeof(float*), sizeof(float*) * hResolution);

	if(cudaSuccess != cudaMallocManaged(&cu_image, sizeof(float*) * hResolution)) //setup 1st dimention input array
	{
		printf("allocation Error\n");
		return 0;
	}
	printf("malloced first\n");

	for(unsigned int i = 0; i < hResolution; i++) // allocate second dimention storage
	{
		cudaMallocManaged(&cu_image[i], sizeof(float) * vResolution);
	}

	printf("cuda image array allocated\n");

	for (unsigned int x = 0; x < hResolution; x++)
	{
		for (unsigned int y = 0; y < vResolution; y++)
		{
			cu_image[x][y] = image[x][y];
		}
		
	}
	

	cudaMallocManaged((void **) &output, sizeof(float*) * hResolution); //setup 1st dimention array

	for(unsigned int i = 0; i < hResolution; i++) // allocate second dimention storage
	{
		cudaMallocManaged(&output[i], sizeof(float) * vResolution);
	}
	printf("cuda outputarr allocated\n");

//	unsigned char exitflag = 0;

	while(index < (vResolution * hResolution))
	{
		printf("%li, ", index);
		fflush(stdout);
		if(((vResolution * hResolution) - index) < (BLOCKS * THREADS))
		{
			ParalellTotalContrastParallelismFunction <<<(((vResolution * hResolution) - index)),(1)>>> (cu_image, output, (radius * hResolution), index, hResolution, vResolution);
			index += ((vResolution * hResolution) - index);
		}
		else
		{
			ParalellTotalContrastParallelismFunction <<<(THREADS),(BLOCKS)>>> (cu_image, output, (radius * hResolution), index, hResolution, vResolution);
			index += (BLOCKS * THREADS);
		}
		printf("\r");
		printStatusBar((unsigned char)(((float)index / (float)(hResolution * vResolution)) * 100));
		cudaDeviceSynchronize();
	}

	for (unsigned long x = 0; x < hResolution; x++)
	{
		for (unsigned long y = 0; y < vResolution; y++)
		{
			returnval += output[x][y];
		}
	}

	for (unsigned int i = 0; i < hResolution; i++)
	{
		cudaFree(output[i]);
	}
	cudaFree(output);

	for (unsigned int i = 0; i < hResolution; i++)
	{
		cudaFree(cu_image[i]);
	}
	cudaFree(cu_image);

	return returnval / (hResolution * vResolution);
}

__global__ void ParalellTotalContrastParallelismFunction (float **InputData, float **outputData, unsigned int frame, unsigned long index, unsigned int hResolution, unsigned int vResolution) // supposed to be run in threads
{
	float arrMax = -100;
	float arrMin = 100;

	unsigned int x = (index + threadIdx.x) % hResolution;
	unsigned int y = (index + threadIdx.x) / hResolution;

	for (signed int testarrX = 0; testarrX < frame; testarrX++)
	{
		for (signed int testarrY = 0; testarrY < frame; testarrY++)
		{
			float tempval = InputData[limit(x + testarrX - (frame / 2), 0, (hResolution - 1))][limit(y + testarrY - (frame / 2), 0, (vResolution - 1))];
			if (tempval < arrMin)
			{
				arrMin = tempval;
			}
			if (tempval > arrMax)
			{
				arrMax = tempval;
			}
		}
	}
	outputData[x][y] = arrMax - arrMin;
}

__device__ static inline signed long limit(signed long input, signed long lower, signed long upper)
{
	if (input > upper)
	{
		return upper;
	}
	if (input < lower)
	{
		return lower;
	}
	else
	{
		return input;
	}
}
