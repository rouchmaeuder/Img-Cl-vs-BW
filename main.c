#include "libs/cuda_acceleration.h"
#include "time.h"
#include "dirent.h"
#include "string.h"
#include "libs/tiff.h"

// compile with gcc main.c -lm -o main.o 
// link with 	gcc main.o libs/cuda_acceleration.o -lcudart -L/usr/local/cuda/lib64 -lm -o a.out
// total command /usr/local/cuda/bin/nvcc /home/user/tiff_file_parser/libs/cuda_acceleration.cu -Xcompiler "-fPIC" -o /home/user/tiff_file_parser/libs/cuda_acceleration.o -c && gcc main.c -lm -o main.o -c && gcc main.o libs/cuda_acceleration.o -lcudart -L/usr/local/cuda/lib64 -lm -lstdc++ -v -o a.out

#define ANSI_RESET "\x1b[0m"
#define ANSI_WHITE_BKGRND "\x1b[30;107m"

#define OUTPUTFILENAME "outputs.csv"

void printStatusBar(unsigned char input);																					   // takes a unsigned char from 0 to 100 as a percentage
static inline signed long limit(signed long input, signed long lower, signed long upper);

float totalContrast(float **image, float radius); // calculate contrast

int main(int argc, char * argv[])
{
	VerboseFlag = PrintNone;

	if(argc <= 1)
	{
		return 1;
	}

	DIR * imgDir = opendir(argv[1]);

	if (imgDir == NULL)
	{
		printf("was not able to open directory\n");
		return 0;
	}
	printf("directory opened\n");

	struct dirent *filepath;
	unsigned int filenum = 0;
	char** filePathArr = NULL;

	filepath = readdir(imgDir);
	while (filepath != NULL)
	{
		char * DotTifPos = strstr(filepath->d_name, ".tif");
		//printf("why %li \n", ((filepath->d_name + strlen(filepath->d_name)) - DotTifPos));
		if((DotTifPos != NULL) && ((filepath->d_name + strlen(filepath->d_name)) - DotTifPos) < 5 && (filepath->d_type == DT_REG)) // search folder for .tif files and store their paths in filePathArr. store the amount of files found in filenum
		{
			if(filenum)
			{
				filePathArr = realloc(filePathArr, sizeof(char*) * (1 + filenum));
			}
			else
			{
				filePathArr = malloc(sizeof(char*));
			}
			
			filePathArr[filenum] = malloc(sizeof(filepath->d_name) + sizeof(argv[1]));
			strcpy(filePathArr[filenum], argv[1]);
			strcat(filePathArr[filenum], filepath->d_name);
			printf("%s\n", filePathArr[filenum]);

			filenum++;
		}
		filepath = readdir(imgDir);
	}

	char* outputFilePath = malloc(sizeof(argv[1]) + sizeof(OUTPUTFILENAME));
	strcpy(outputFilePath, argv[1]);
	strcat(outputFilePath, OUTPUTFILENAME);
	FILE *outFile = fopen(outputFilePath, "w+"); // create output .csv file
	free(outputFilePath);

	struct tiff img;
	for (unsigned char i = 0; i < filenum; i++) // loop over all filepaths
	{
		openTiff(&img, 1, filePathArr[i]); // open the image
		float contrastVal = ParalellTotalContrast(img.BW_Data, 0.001, img.vResolution, img.hResolution); // analyze the image
		closeTiff(&img); // close the image

		printf("total contrast is %f\n", contrastVal); // print the resolution to console
		fprintf(outFile,"%s, %f \n", filePathArr[i], contrastVal); // print result to file
		free(filePathArr[i]); // free the filepath array
	}

	fclose(outFile);
	free(filePathArr);
	closedir(imgDir);
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
