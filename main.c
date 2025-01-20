#include "libs/cuda_acceleration.h"
#include "time.h"
#include "dirent.h"
#include "string.h"
#include "libs/tiff.h"
#include <gtk/gtk.h>

// compile with gcc main.c -lm -o main.o 
// link with 	gcc main.o libs/cuda_acceleration.o -lcudart -L/usr/local/cuda/lib64 -lm -o a.out
// total command /usr/local/cuda/bin/nvcc /home/user/tiff_file_parser/libs/cuda_acceleration.cu -Xcompiler "-fPIC" -o /home/user/tiff_file_parser/libs/cuda_acceleration.o -c && gcc main.c -lm -o main.o -c && gcc main.o libs/cuda_acceleration.o -lcudart -L/usr/local/cuda/lib64 -lm -lstdc++ -v -o a.out

#define ANSI_RESET "\x1b[0m"
#define ANSI_WHITE_BKGRND "\x1b[30;107m"

#define PREVIEWWINDOWRES 500

#define OUTPUTFILENAME "outputs.csv"

void printStatusBar(unsigned char input);																					   // takes a unsigned char from 0 to 100 as a percentage
static inline signed long limit(signed long input, signed long lower, signed long upper);

float totalContrast(float **image, float radius); // calculate contrast

struct tiff * currentImgPtr;

static void openWindow (GtkApplication * app);

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
		currentImgPtr = &img;
		g_autoptr(GtkApplication) imgPrev = gtk_application_new(NULL, 0);
		g_signal_connect(imgPrev, "activate", G_CALLBACK(openWindow), NULL);
		g_application_run (G_APPLICATION (imgPrev), /*argc*/ 0, /*argv*/ 0);

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

static void openWindow (GtkApplication * app)
{
	GtkWindow * window;
	GtkWidget * image;
	unsigned char * arr;
	unsigned int i = 0;
	float max = 0;
	float min = 1000;
	window = GTK_WINDOW(gtk_application_window_new(app));
	image = gtk_picture_new();
	gtk_window_set_child(window, image);
	
	arr = malloc(PREVIEWWINDOWRES * PREVIEWWINDOWRES * 3);
	uint32_t yprevres = (float)PREVIEWWINDOWRES / ((float)currentImgPtr->hResolution/(float)currentImgPtr->vResolution);
	for (uint32_t y = 0; y < yprevres; y++)
	{
		for (uint32_t x = 0; x < PREVIEWWINDOWRES; x++)
		{
			uint32_t xreadout = (((float)x+0.5)/(float)PREVIEWWINDOWRES)*currentImgPtr->hResolution;
			uint32_t yreadout = (((float)y+0.5)/(float)PREVIEWWINDOWRES)*currentImgPtr->hResolution;
			if (xreadout > currentImgPtr->hResolution || yreadout > currentImgPtr->vResolution)
			{
				printf("x= %i, y= %i;	hres= %li, vres= %li \n", xreadout, yreadout, currentImgPtr->hResolution, currentImgPtr->vResolution);
			}
			else
			{
				for (unsigned char color = 0; color < 3; color++)
				{
					float temp = currentImgPtr->RGB_Data[color][xreadout][yreadout];
				
					if (temp > max)
					{
						max = temp;
					}
					if (temp < min)
					{
						min = temp;
					}
					
					arr[i] = limit((int)((temp) * 254.9), 0, 0xff);
					i++;
				}
			}
		}
	}

	printf("max = %f min = %f \n", max, min);
	
	gtk_picture_set_paintable(image, GDK_PAINTABLE(gdk_memory_texture_new(PREVIEWWINDOWRES, PREVIEWWINDOWRES, GDK_MEMORY_R8G8B8, g_bytes_new(arr, PREVIEWWINDOWRES*PREVIEWWINDOWRES*3),PREVIEWWINDOWRES*3)));
	free(arr);

	gtk_window_present(window);
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
