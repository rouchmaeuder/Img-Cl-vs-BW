#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#include "cuda_acceleration.h"

// compile with idfknow nvcc odr so

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
}

float ParalellTotalContrast(float **image, float radius, unsigned int vResolution, unsigned int hResolution)
{
	unsigned int frame = radius * hResolution;
	float returnval = 0;
	printf("calculating avg contrast: \n");

	unsigned long index = 0;

	float ** cu_image;
	float ** output;


	cudaMallocManaged((void **) &cu_image, sizeof(float*) * hResolution); //setup 1st dimention input array

	for(unsigned int i = 0; i < hResolution; i++) // allocate second dimention storage
	{
		cudaMallocManaged(&cu_image[i], sizeof(float) * vResolution);
	}

	printf("cuda image array allocated");

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
	printf("cuda outputarr allocated");

	while(index < (vResolution * hResolution))
	{
		if(((vResolution * hResolution) - index) < 512)
		{
			ParalellTotalContrastParallelismFunction <<<(((vResolution * hResolution) - index)),(1)>>> (cu_image, output, (radius * hResolution), index, hResolution, vResolution);
			index += ((vResolution * hResolution) - index);
		}
		else
		{
			ParalellTotalContrastParallelismFunction <<<(512),(1)>>> (cu_image, output, (radius * hResolution), index, hResolution, vResolution);
			index += 512;
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