#include "libs/cuda_acceleration.h"
#include "time.h"
#include "libs/tiff.h"

// compile with gcc main.c -lm -o main.o 
// link with 	gcc main.o libs/cuda_acceleration.o -lcudart -L/usr/local/cuda/lib64 -lm -o a.out
// total command /usr/local/cuda/bin/nvcc /home/user/tiff_file_parser/libs/cuda_acceleration.cu -Xcompiler "-fPIC" -o /home/user/tiff_file_parser/libs/cuda_acceleration.o -c && gcc main.c -lm -o main.o -c && gcc main.o libs/cuda_acceleration.o -lcudart -L/usr/local/cuda/lib64 -lm -lstdc++ -v -o a.out

#define ANSI_RESET "\x1b[0m"
#define ANSI_WHITE_BKGRND "\x1b[30;107m"

void printStatusBar(unsigned char input);																					   // takes a unsigned char from 0 to 100 as a percentage
static inline signed long limit(signed long input, signed long lower, signed long upper);

float totalContrast(float **image, float radius); // calculate contrast

int main(void)
{
	VerboseFlag = 0;
	struct tiff img;
	if (openTiff(&img, 1, "/home/user/tiff_file_parser/data/pic.tif") != Success)
	{
		printf("error\n");
		return 1;
	}

	printf("evaluating with graphicscard\n");
	time_t seconds = time(NULL);
	printf("total contrast is %f\n", ParalellTotalContrast(img.BW_Data, 0.01, img.vResolution, img.hResolution));
	time_t seconds_ref = time(NULL);
	printf("took %li seconds\n", seconds_ref - seconds);

	printPreview(&img, 32);

	closeTiff(&img);
	return 0;
}


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

/*float totalContrast(float **image, float radius)
{
	unsigned int frame = radius * hResolution;
	float returnval = 0;
	float arrMax = 0;
	float arrMin = 0;
	printf("calculating avg contrast: \n");
	for (unsigned long x = 0; x < hResolution; x++)
	{
		for (unsigned long y = 0; y < vResolution; y++)
		{
			// loops over every pixel
			for (signed int testarrX = 0; testarrX < frame; testarrX++)
			{
				for (signed int testarrY = 0; testarrY < frame; testarrY++)
				{
					float tempval = image[limit(x + testarrX - (frame / 2), 0, (hResolution - 1))][limit(y + testarrY - (frame / 2), 0, (vResolution - 1))];
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
			returnval += (arrMax - arrMin);
			arrMax = -100;
			arrMin = 100;
		}
		printf("\r");
		printStatusBar((unsigned char)(((float)x / (float)hResolution) * 100));
	}
	return returnval / (hResolution * vResolution);
}*/

static inline signed long limit(signed long input, signed long lower, signed long upper)
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
