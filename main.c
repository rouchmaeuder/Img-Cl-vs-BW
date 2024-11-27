#include "stdlib.h"
#include "stdio.h"
#include "math.h"

// compile with gcc -g main.c -lm

#define BYTE_TYPE 1
#define ASCII_TYPE 2
#define INT_TYPE 3
#define LONG_TYPE 4
#define RATIO_TYPE 5

#define ANSI_RESET "\x1b[0m"
#define ANSI_WHITE_BKGRND "\x1b[30;107m"

#define RED 0
#define GREEN 1
#define BLUE 2

#define PREVRES 32

#define VERBOSE

enum data_units
{
	BYTE = 1,
	CHAR = 2,
	SHORT = 3,
	LONG = 4,
	RATIONAL = 5
};

struct IFD_Entry
{
	unsigned int tag;
	unsigned int type;
	unsigned long data_arr_len;
	unsigned long data_offset;
};

enum compression_Type
{
	backTobackData = 1,
	huffmanRunlength = 2,
	bitPacking = 32775
};

const unsigned char data_unit_len[6] =
	{
		0,
		1,
		1,
		2,
		4,
		8};

struct IFD_Entry *IFD_arr_ptr;

unsigned int ***dataStripPointer = NULL;

enum compression_Type compression_Type = 0;

unsigned int IFD_entries;

unsigned long IFD_index_offset = 0;

unsigned long hResolution = 0;
unsigned long vResolution = 0;
unsigned int SamplesPerPixel = 0;

unsigned int errorflags = 0;
/*
bit  |error
0x01 |file could not be opened
0x02 |file does not have necessary 42 at ptr+2
0x04 |file is in big endian
0x08 |unsupported encoding type
0x10 |above 1 img strip
*/

FILE *fptr;

float ***pixelData = NULL;

unsigned char freadchar(FILE *file, unsigned long offset);																	   // Reads a unsigned char out of a file at a specified offset within
unsigned int freadint(FILE *file, unsigned long offset);																	   // Reads a unsigned integer out of a file at a specified offset within
unsigned long freadlong(FILE *file, unsigned long offset);																	   // Reads a unsigned long out of a file at a specified offset within
unsigned long IFDReadInteger(FILE *file, struct IFD_Entry Entry);															   // Reads up to a unsigned long out of an IFD
unsigned long IFDReadEntry(FILE *file, struct IFD_Entry *IFD_entries_arr_ptr, unsigned int IFD_Entry_count, unsigned int tag); // Reads up to a unsigned long out of an IFD with a specified tag out of an array of IFD_entry types
unsigned long fillLongInt(FILE *file, unsigned long ZoneOffset, unsigned long bitOffset, unsigned char bitlen);				   // deprecated method to uncompress back to back data compression
void printStatusBar(unsigned char input);																					   // takes a unsigned char from 0 to 100 as a percentage
signed long limit(signed long input, signed long lower, signed long upper);

float totalContrast(float **image, float radius); // calculate contrast

void main(void)
{
	fptr = fopen("/home/user/tiff_file_parser/pic.tif", "r");

	if (fptr == NULL) // check for file opening errors
	{
		printf("was not able to open file \n");
		errorflags |= 0x01;
	}

	if (!(errorflags)) // check for endianness (necessary to check for tiff flag as it is influenced by endianness too)
	{
		printf("file opened \n");
		if (getc(fptr) != 0x49)
		{
			printf("file is in big endian. this is currently not supported \n");
			errorflags |= 0x04;
		}
	}

	if (errorflags & ~0x0001) // check for tiff/dnf flag
	{
		if (freadchar(fptr, 2))
		{
			printf("this file does not conform to tiff/dnf \n");
			errorflags |= 0x02;
		}
	}

	if (!(errorflags)) // parse IFD length and allocate memory to mirror
	{
		for (unsigned char i = 0; i < 4; i++)
		{
			IFD_index_offset |= (freadchar(fptr, (4 + i)) << (i * 8));
		}

		IFD_entries = freadint(fptr, IFD_index_offset);

		IFD_arr_ptr = (struct IFD_Entry *)malloc(sizeof(struct IFD_Entry) * freadchar(fptr, IFD_index_offset));

		if (IFD_arr_ptr == NULL)
		{
			printf("IFD arr memory could not be allocated \n");
			errorflags |= 0x08;
		}
	}

	if (!(errorflags)) // parse IFD and mirror tag datatype and datalocation
	{
		for (unsigned int i = 0; i < IFD_entries; i++)
		{
			IFD_arr_ptr[i].tag = freadint(fptr, IFD_index_offset + 2 + (12 * i));

			IFD_arr_ptr[i].type = freadint(fptr, IFD_index_offset + 4 + (12 * i));

			IFD_arr_ptr[i].data_arr_len = freadlong(fptr, IFD_index_offset + 6 + (12 * i));

			if (data_unit_len[IFD_arr_ptr[i].type] * IFD_arr_ptr[i].data_arr_len <= 4)
			{
				IFD_arr_ptr[i].data_offset = IFD_index_offset + 10 + (12 * i);
			}
			else
			{
				IFD_arr_ptr[i].data_offset = freadlong(fptr, IFD_index_offset + 10 + (12 * i));
			}
		}
	}

#ifdef VERBOSE
	if (!(errorflags)) // parse and print IFD entries
	{
		for (unsigned int i = 0; i < IFD_entries; i++)
		{
			printf("IFD entry Nr. %i \n", i);
			printf("IFD tag %i \n", IFD_arr_ptr[i].tag);
			printf("array is %li long\n", IFD_arr_ptr[i].data_arr_len);
			printf("datatype is");

			if (IFD_arr_ptr[i].type == 2 || (IFD_arr_ptr[i].data_arr_len >= 3 && IFD_arr_ptr[i].data_arr_len <= 50)) // check that string is between 3 and 50 chars long (this is to not print possible classification letters or long html stuff)
			{
				char string[(IFD_arr_ptr[i].data_arr_len) + 1];
				for (unsigned int y = 0; y < IFD_arr_ptr[i].data_arr_len; y++)
				{
					string[y] = freadchar(fptr, IFD_arr_ptr[i].data_offset + y);
				}
				string[IFD_arr_ptr[i].data_arr_len] = '\0';
				printf(" string with contents: ");
				printf("%s", string);
				printf("\n");
			}
			else
			{
				printf(" 0x%04x ", IFD_arr_ptr[i].type);
				printf("at adress 0x%08lx \n", IFD_arr_ptr[i].data_offset);
			}

			printf("\n");
		}
	}
#endif

	if (!(errorflags)) // print data compression
	{
		for (unsigned int i = 0; i < IFD_entries; i++)
		{
			if (IFD_arr_ptr[i].tag == 259) // if tag == compression type
			{
				compression_Type = freadint(fptr, IFD_arr_ptr[i].data_offset);
				i = 0xfffe; // for exit statement
				switch (compression_Type)
				{
				case backTobackData:
					printf("encoding type is backtoback data. \nthis file is readable \n");
					break;

				case huffmanRunlength:
					printf("this file is huffman encoded and tha data is thus not supported");
					errorflags |= 0x08;
					break;

				case bitPacking:
					printf("the encoding type bitpacking encoding is currently not supported");
					errorflags |= 0x08;
					break;

				default:
					printf("compression type unknown. compression type nr is 0x%04x \n", compression_Type);
					break;
				}
			}
		}
	}

	if (!(errorflags)) // parse data strippattern
	{
		unsigned int stripCount = 0;
		unsigned int BitsPerSample = (unsigned int)IFDReadEntry(fptr, IFD_arr_ptr, IFD_entries, 258);
		unsigned long hLinesPerStrip = IFDReadEntry(fptr, IFD_arr_ptr, IFD_entries, 278); // tag 278
		unsigned long StripOffsets = IFDReadEntry(fptr, IFD_arr_ptr, IFD_entries, 273);	  // tag 273
		unsigned long StripByteCount = IFDReadEntry(fptr, IFD_arr_ptr, IFD_entries, 279); // tag 279
		SamplesPerPixel = IFDReadEntry(fptr, IFD_arr_ptr, IFD_entries, 277);
		hResolution = IFDReadEntry(fptr, IFD_arr_ptr, IFD_entries, 256);
		vResolution = IFDReadEntry(fptr, IFD_arr_ptr, IFD_entries, 257);
		unsigned int perRowBytes = (hResolution * SamplesPerPixel * BitsPerSample / 8);

		for (unsigned int i = 0; i < IFD_entries; i++)
		{
			if (IFD_arr_ptr[i].tag == 273)
			{
				stripCount = IFD_arr_ptr[i].data_arr_len;
			}
		}

#ifdef VERBOSE
		printf("the image is %li", hResolution);
		printf("x %li pixels large ", vResolution);
		printf("and split into %i chunk(s) \n", stripCount);
#endif
		printf("bits per sample 0x%04x \n", BitsPerSample);
		printf("with %i samples per pixel\n", SamplesPerPixel);

		if (stripCount > 1)
		{
			printf("more than 1 strip not supported");
			errorflags |= 0x10;
		}
		else
		{
			pixelData = (float ***)malloc(sizeof(unsigned long **) * SamplesPerPixel);
			for (unsigned int i = 0; i < SamplesPerPixel; i++)
			{
				pixelData[i] = (float **)malloc(sizeof(unsigned long *) * hResolution);
				for (unsigned int j = 0; j < hResolution; j++)
				{
					pixelData[i][j] = (float *)malloc(sizeof(unsigned long) * vResolution);
				}
			}

#ifdef VERBOSE
			printf("data allocated and arr structure written");
#endif
			printf("\n");

			unsigned char *temp_imgData = malloc((perRowBytes * vResolution * sizeof(unsigned char)) + 1);

			fseek(fptr, StripOffsets, SEEK_SET);
			fread(temp_imgData, sizeof(unsigned char), (perRowBytes * vResolution) + 1, fptr);

			printf("temparr initialized and filled");

			unsigned long counter = 0;

			for (unsigned long x = 0; x < hResolution; x++)
			{
				for (unsigned long y = 0; y < vResolution; y++)
				{
					for (unsigned char sample = 0; sample < SamplesPerPixel; sample++)
					{
						unsigned int arrptr = (perRowBytes * y) + (((x * BitsPerSample * SamplesPerPixel) + (sample * BitsPerSample)) / 8);
						unsigned long LeftJustifiedSubpixelLuminocity = temp_imgData[arrptr] << ((((x * SamplesPerPixel * BitsPerSample) + (sample * SamplesPerPixel)) % 8) + 24);
						for (unsigned int i = 1; i < (BitsPerSample / 8) + 1; i++)
						{
							LeftJustifiedSubpixelLuminocity |= temp_imgData[arrptr + i] << 32 - ((i + 1) * 8);
						}

						LeftJustifiedSubpixelLuminocity &= (0xffffffff << (32 - BitsPerSample));
						pixelData[sample][x][y] = (float)LeftJustifiedSubpixelLuminocity / (float)0xffffffff;
					}
				}
				printf("\r");
				printf("parsing imgData ");
				printStatusBar((unsigned char)(((float)x / (float)hResolution) * 100));
			}
			printf("\n");
			free(temp_imgData);
		}
	}

	if (!(errorflags)) // print preview image
	{
		unsigned int PixelsPerPixelY = vResolution / PREVRES;
		unsigned int PixelsPerPixelX = PixelsPerPixelY / 2;
		unsigned int previewXRes = hResolution / PixelsPerPixelX;
		unsigned long divBy = PixelsPerPixelX * PixelsPerPixelY * SamplesPerPixel;
		float pixelavg = 0;
		float totalpixelavg = 0;
		unsigned long counter = 0;

#ifdef VERBOSE
		printf("a preview pixel is composed of %i x %i\n", PixelsPerPixelX, PixelsPerPixelY);
		printf("this results in a image wich is %i x %i\n", PREVRES, (unsigned int)previewXRes);
		printf("thus each subpixels has to be divided by %li to summed together result in equal weights\n", divBy);
#endif

		for (unsigned int x = 0; x < hResolution; x++)
		{
			for (unsigned int y = 0; y < vResolution; y++)
			{
				float tempval = 0;
				for (unsigned int Sample = 0; Sample < SamplesPerPixel; Sample++)
				{
					tempval += powf(pixelData[Sample][x][y], 2);
				}
				totalpixelavg += (sqrtf(tempval)) / (hResolution * vResolution);
			}
		}

		printf("the total average luminocity of the image is %f \n", totalpixelavg);

		for (unsigned int y = 0; y < PREVRES; y++)
		{
			for (unsigned int x = 0; x < previewXRes; x++)
			{
				for (unsigned int orgX = (x * PixelsPerPixelX); (orgX < (PixelsPerPixelX * (x + 1)) && orgX < hResolution); orgX++)
				{
					for (unsigned int orgY = (y * PixelsPerPixelY); (orgY < (PixelsPerPixelY * (y + 1)) && orgY < vResolution); orgY++)
					{
						float tempval = 0;
						for (unsigned int Sample = 0; Sample < SamplesPerPixel; Sample++)
						{
							tempval += powf(pixelData[Sample][orgX][orgY], 2);
						}
						pixelavg += sqrtf(tempval);
					}
				}
				if ((pixelavg / (PixelsPerPixelX * PixelsPerPixelY)) > (totalpixelavg))
				{
					printf(ANSI_WHITE_BKGRND " " ANSI_RESET);
				}
				else
				{
					printf(" ");
				}
				pixelavg = 0;
			}
			printf("\n");
		}
	}

	if (!(errorflags)) // evaluate
	{
		//		printf("total contrast is %f\n", totalContrast(pixelData[1], 0.1));
	}

	if (!(errorflags & ~0x01)) // freeing memory
	{
		free(IFD_arr_ptr);

		for (unsigned int i = 0; i < SamplesPerPixel; i++)
		{
			for (unsigned long j = 0; j < hResolution; j++)
			{
				free((void *)pixelData[i][j]);
			}
			free((void *)pixelData[i]);
		}
		free((void *)pixelData);

		fclose(fptr);
	}

	printf("program ended with exitcode 0x%04x \n", errorflags);
}

unsigned char freadchar(FILE *file, unsigned long offset)
{
	fseek(file, offset, SEEK_SET);
	return getc(file);
}

unsigned int freadint(FILE *file, unsigned long offset)
{
	return (unsigned int)freadchar(file, offset) | ((unsigned int)freadchar(file, offset + 1) << 8);
}

unsigned long freadlong(FILE *file, unsigned long offset)
{
	return (((unsigned long)freadchar(file, offset)) |
			((unsigned int)freadchar(file, offset + 1) << 8) |
			((unsigned int)freadchar(file, offset + 2) << 16) |
			((unsigned int)freadchar(file, offset + 3) << 24));
}

unsigned long IFDReadInteger(FILE *file, struct IFD_Entry Entry)
{
	if (Entry.type == BYTE_TYPE)
	{
		return (unsigned long)freadchar(file, Entry.data_offset);
	}
	if (Entry.type == INT_TYPE)
	{
		return (unsigned long)freadint(file, Entry.data_offset);
	}
	if (Entry.type == LONG_TYPE)
	{
		return (unsigned long)freadlong(file, Entry.data_offset);
	}

	return 0;
}

unsigned long IFDReadEntry(FILE *file, struct IFD_Entry *IFD_entries_arr_ptr, unsigned int IFD_Entry_count, unsigned int tag)
{
	for (unsigned int i = 0; i < IFD_Entry_count; i++)
	{
		if (IFD_entries_arr_ptr[i].tag == tag)
		{
			return IFDReadInteger(file, IFD_entries_arr_ptr[i]);
		}
	}
	return 0;
}

unsigned long fillLongInt(FILE *file, unsigned long ZoneOffset, unsigned long bitOffset, unsigned char bitlen)
{
	unsigned long tempval = 0;
	for (unsigned char i = 0; i < 5; i++)
	{
		tempval |= (freadchar(file, ZoneOffset + (bitOffset / 8) + i)) << ((24 - (i * 8)) + (32 - bitlen));
	}
	return tempval | ~(0xffffffff << bitlen);
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
}

float totalContrast(float **image, float radius)
{
	unsigned int frame = radius * hResolution;
	float returnval = 0;
	for (unsigned long x = 0; x < hResolution; x++)
	{
		for (unsigned long y = 0; y < vResolution; y++)
		{
			// loops over every pixel
			float arrMax = -100;
			float arrMin = 100;
			for (signed int testarrX = -(frame / 2); testarrX < (frame / 2); testarrX++)
			{
				for (signed int testarrY = -(frame / 2); testarrY < (frame / 2); testarrY++)
				{
					float tempval = image[limit(x + testarrX, 0, hResolution)][limit(y + testarrY, 0, vResolution)];
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
		}
	}
	return returnval / (hResolution * vResolution);
}

signed long limit(signed long input, signed long lower, signed long upper)
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
