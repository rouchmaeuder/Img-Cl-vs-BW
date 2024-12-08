#include "libs/cuda_acceleration.h"
#include "time.h"
#include "dirent.h"
#include "libs/tiff.h"

// compile with gcc main.c -lm -o main.o 
// link with 	gcc main.o libs/cuda_acceleration.o -lcudart -L/usr/local/cuda/lib64 -lm -o a.out
// total command /usr/local/cuda/bin/nvcc /home/user/tiff_file_parser/libs/cuda_acceleration.cu -Xcompiler "-fPIC" -o /home/user/tiff_file_parser/libs/cuda_acceleration.o -c && gcc main.c -lm -o main.o -c && gcc main.o libs/cuda_acceleration.o -lcudart -L/usr/local/cuda/lib64 -lm -lstdc++ -v -o a.out

#define ANSI_RESET "\x1b[0m"
#define ANSI_WHITE_BKGRND "\x1b[30;107m"

void printStatusBar(unsigned char input);																					   // takes a unsigned char from 0 to 100 as a percentage
static inline signed long limit(signed long input, signed long lower, signed long upper);

float totalContrast(float **image, float radius); // calculate contrast

int main(int argc, char * argv[])
{
	VerboseFlag = PrintNone;
/*	DIR * imgDir = opendir(argv[1]);

	if (imgDir == NULL)
	{
		printf("was not able to open directory");
		return 0;
	}*/
	

	struct tiff img;
	enum errorType dbgError = openTiff(&img, 1, "/home/user/tiff_file_parser/data/pic.tif");
	if (dbgError != Success)
	{
		printf("error%s\n", errConvertToString(dbgError));
		return 1;
	}

	printf("evaluating with graphicscard\n");
	time_t seconds = time(NULL);
//	printf("total contrast is %f\n", ParalellTotalContrast(img.BW_Data, 0.01, img.vResolution, img.hResolution));
	time_t seconds_ref = time(NULL);
	printf("took %li seconds\n", seconds_ref - seconds);

	printPreview(&img, 32);

	closeTiff(&img);
//	closedir(imgDir);
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
