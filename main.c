#include "libs/cuda_acceleration.h"
#include "time.h"
#include "dirent.h"
#include "string.h"
#include "libs/tiff.h"


#define ANSI_RESET "\x1b[0m"
#define ANSI_WHITE_BKGRND "\x1b[30;107m"

#define PREVIEWWINDOWRES 500

#define OUTPUTFILENAME "outputs.csv"

void printStatusBar(unsigned char input);	  // takes a unsigned char from 0 to 100 as a percentage
static inline signed long limit(signed long input, signed long lower, signed long upper);

float totalContrast(float **image, float radius); // calculate contrast

struct tiff * currentImgPtr;


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
		if((DotTifPos != NULL) && ((filepath->d_name + strlen(filepath->d_name)) - DotTifPos) < 6 && (filepath->d_type == DT_REG)) // search folder for .tif files and store their paths in filePathArr. store the amount of files found in filenum
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

	printf("files indexed \n");

	char* outputFilePath = malloc(strlen(argv[1]) + strlen(OUTPUTFILENAME) + 1);
	strcpy(outputFilePath, argv[1]);
	strcat(outputFilePath, OUTPUTFILENAME);
	FILE *outFile = fopen(outputFilePath, "w+"); // create or !!overwrite!! output .csv file
	free(outputFilePath);

	printf("outputfile created\n");
	printf("\n  ");

	struct tiff img;

	for (unsigned char i = 0; i < filenum; i++) // loop over all filepaths
	{
		if(openTiff(&img, 1, filePathArr[i]) != 0) // open the image
		{
			printf("error on file %i, %s\n", i, filePathArr[i]);
		} else {
			currentImgPtr = &img;
			fprintf(outFile,"%s", filePathArr[i]);
			for (__uint16_t j = 0; j < 8; j++)
			{
				printf("\r \bfile %i of %i, pass %i of 8\n", i, filenum, j);
				float contrastVal = ParalellTotalContrast(img.BW_Data, powf(2, ((0-4)-j)), img.vResolution, img.hResolution); // analyze the image
				fprintf(outFile, ", %f", contrastVal);
			}
			fprintf(outFile,"\n");

			currentImgPtr = &img;
			
			closeTiff(&img); // close the image
			
			free(filePathArr[i]); // free the filepath array
		}
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
