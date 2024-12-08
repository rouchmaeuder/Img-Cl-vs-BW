#include "tiff.h"

#define ANSI_RESET "\x1b[0m"
#define ANSI_WHITE_BKGRND "\x1b[30;107m"

#define ERRORS 0x01
#define STATS 0x02
#define STATUSBAR 0x04
#define DEBUGINFOS 0x80

const unsigned char data_unit_len[6] = {
	0,
	1,
	1,
	2,
	4,
	8
};

enum compression_Type
{
	Null = 0,
	backTobackData = 1,
	huffmanRunlength = 2,
	bitPacking = 32775
};

enum verbosityLevel VerboseFlag = 0;

static unsigned char freadchar(FILE *file, unsigned long offset);																	   // Reads a unsigned char out of a file at a specified offset within
static unsigned int freadint(FILE *file, unsigned long offset);																	   // Reads a unsigned integer out of a file at a specified offset within
static unsigned long freadlong(FILE *file, unsigned long offset);																	   // Reads a unsigned long out of a file at a specified offset within
static unsigned long IFDReadInteger(FILE *file, struct IFD_Entry Entry);															   // Reads up to a unsigned long out of an IFD
static unsigned long IFDReadEntry(FILE *file, struct IFD_Entry *IFD_entries_arr_ptr, unsigned int IFD_Entry_count, unsigned int tag); // Reads up to a unsigned long out of an IFD with a specified tag out of an array of IFD_entry types
static unsigned long fillLongInt(FILE *file, unsigned long ZoneOffset, unsigned long bitOffset, unsigned char bitlen);				   // deprecated method to uncompress back to back data compression
static void printStatusBar(unsigned char input);	

enum errorType openTiff(struct tiff* imgObj, unsigned char ConstructBwFlag, char filePath [])
{
    enum errorType errorflags;
    unsigned int IFD_index_offset;

	imgObj->OriginFilePointer = fopen(filePath, "r");

	if (imgObj->OriginFilePointer == NULL) // check for file opening errors
	{
		if (VerboseFlag & ERRORS)
		{
			printf("was not able to open file \n");
		}
		errorflags |= 0x01;
	}

	if (!(errorflags)) // check for endianness (necessary to check for tiff flag as it is influenced by endianness too)
	{
		if (VerboseFlag & DEBUGINFOS)
		{
			printf("file opened \n");
		}
		if (getc(imgObj->OriginFilePointer) != 0x49)
		{
			if (VerboseFlag & ERRORS)
			{
				printf("file is in big endian. this is currently not supported \n");
			}
			errorflags |= 0x04;
		}
	}

	if (!(errorflags)) // check for tiff/dnf flag
	{
		if (freadchar(imgObj->OriginFilePointer, 2) != 42)
		{
			if(VerboseFlag & ERRORS)
			{
				printf("this file does not conform to tiff/dnf \n");
			}
			errorflags |= 0x02;
		}
	}

	if (!(errorflags)) // parse IFD length and allocate memory to mirror
	{
		for (unsigned char i = 0; i < 4; i++)
		{
			IFD_index_offset |= (freadchar(imgObj->OriginFilePointer, (4 + i)) << (i * 8));
		}

		imgObj->IFD.Entries = freadint(imgObj->OriginFilePointer, IFD_index_offset);

        imgObj->IFD.IFDs = (struct IFD_Entry *)malloc(sizeof(struct IFD_Entry) * freadchar(imgObj->OriginFilePointer, IFD_index_offset));

		if (imgObj->IFD.IFDs == NULL)
		{
			if(VerboseFlag & ERRORS)
			{
				printf("IFD arr memory could not be allocated \n");
			}
			errorflags |= 0x08;
		}
	}

	if (!(errorflags)) // parse IFD and mirror tag datatype and datalocation
	{
		for (unsigned int i = 0; i < imgObj->IFD.Entries; i++)
		{
			imgObj->IFD.IFDs[i].tag = freadint(imgObj->OriginFilePointer, IFD_index_offset + 2 + (12 * i));

			imgObj->IFD.IFDs[i].type = freadint(imgObj->OriginFilePointer, IFD_index_offset + 4 + (12 * i));

			imgObj->IFD.IFDs[i].data_arr_len = freadlong(imgObj->OriginFilePointer, IFD_index_offset + 6 + (12 * i));

			if (data_unit_len[imgObj->IFD.IFDs[i].type] * imgObj->IFD.IFDs[i].data_arr_len <= 4)
			{
				imgObj->IFD.IFDs[i].data_offset = IFD_index_offset + 10 + (12 * i);
			}
			else
			{
				imgObj->IFD.IFDs[i].data_offset = freadlong(imgObj->OriginFilePointer, IFD_index_offset + 10 + (12 * i));
			}
		}
	}

	if ((!(errorflags)) && (VerboseFlag & STATS)) // parse and print IFD entries
	{
		for (unsigned int i = 0; i < imgObj->IFD.Entries; i++)
		{
			printf("IFD entry Nr. %i \n", i);
			printf("IFD tag %i \n", imgObj->IFD.IFDs[i].tag);
			printf("array is %li long\n", imgObj->IFD.IFDs[i].data_arr_len);
			printf("datatype is");

			if (imgObj->IFD.IFDs[i].type == 2 || (imgObj->IFD.IFDs[i].data_arr_len >= 3 && imgObj->IFD.IFDs[i].data_arr_len <= 50)) // check that string is between 3 and 50 chars long (this is to not print possible classification letters or long html stuff)
			{
				char string[(imgObj->IFD.IFDs[i].data_arr_len) + 1];
				for (unsigned int y = 0; y < imgObj->IFD.IFDs[i].data_arr_len; y++)
				{
					string[y] = freadchar(imgObj->OriginFilePointer, imgObj->IFD.IFDs[i].data_offset + y);
				}
				string[imgObj->IFD.IFDs[i].data_arr_len] = '\0';
				printf(" string with contents: ");
				printf("%s", string);
				printf("\n");
			}
			else
			{
				printf(" 0x%04x ", imgObj->IFD.IFDs[i].type);
				printf("at adress 0x%08lx \n", imgObj->IFD.IFDs[i].data_offset);
			}

			printf("\n");
		}
	}

	if (!(errorflags)) // print data compression
	{
        enum compression_Type compression = IfdReadInt(imgObj, 259);
		if(VerboseFlag & ERRORS)
		{
	        switch (compression)
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
	            printf("compression type unknown. compression type nr is 0x%04x \n", compression);
				errorflags |= 0x08;
	            break;
        	}
		}
		else
		{
			if (compression != backTobackData)
			{
				errorflags |= 0x08;
			}
		}
	}

	if (!(errorflags)) // parse data strippattern
	{
		unsigned int stripCount = 0;
		unsigned int BitsPerSample = (unsigned int)IFDReadEntry(imgObj->OriginFilePointer, imgObj->IFD.IFDs, imgObj->IFD.Entries, 258);
		unsigned long hLinesPerStrip = IFDReadEntry(imgObj->OriginFilePointer, imgObj->IFD.IFDs, imgObj->IFD.Entries, 278); // tag 278
		unsigned long StripOffsets = IFDReadEntry(imgObj->OriginFilePointer, imgObj->IFD.IFDs, imgObj->IFD.Entries, 273);	  // tag 273
		unsigned long StripByteCount = IFDReadEntry(imgObj->OriginFilePointer, imgObj->IFD.IFDs, imgObj->IFD.Entries, 279); // tag 279
		imgObj->SamplesPerPixel = IFDReadEntry(imgObj->OriginFilePointer, imgObj->IFD.IFDs, imgObj->IFD.Entries, 277);
		imgObj->hResolution = IFDReadEntry(imgObj->OriginFilePointer, imgObj->IFD.IFDs, imgObj->IFD.Entries, 256);
		imgObj->vResolution = IFDReadEntry(imgObj->OriginFilePointer, imgObj->IFD.IFDs, imgObj->IFD.Entries, 257);
		unsigned int perRowBytes = (imgObj->hResolution * imgObj->SamplesPerPixel * BitsPerSample / 8);

		for (unsigned int i = 0; i < imgObj->IFD.Entries; i++)
		{
			if (imgObj->IFD.IFDs[i].tag == 273)
			{
				stripCount = imgObj->IFD.IFDs[i].data_arr_len;
			}
		}

        if (VerboseFlag & STATS)
        {
		    printf("the image is %li", imgObj->hResolution);
		    printf("x %li pixels large ", imgObj->vResolution);
		    printf("and split into %i chunk(s) \n", stripCount);
			printf("bits per sample 0x%04x \n", BitsPerSample);
			printf("with %i samples per pixel\n", imgObj->SamplesPerPixel);
		}

		if (stripCount > 1)
		{
			if(VerboseFlag & ERRORS)
			{
				printf("more than 1 strip not supported");
			}
			errorflags |= 0x10;
		}
		else
		{
			imgObj->RGB_Data = (float ***)malloc(sizeof(unsigned long **) * imgObj->SamplesPerPixel);
			for (unsigned int i = 0; i < imgObj->SamplesPerPixel; i++)
			{
				imgObj->RGB_Data[i] = (float **)malloc(sizeof(unsigned long *) * imgObj->hResolution);
				for (unsigned int j = 0; j < imgObj->hResolution; j++)
				{
					imgObj->RGB_Data[i][j] = (float *)malloc(sizeof(unsigned long) * imgObj->vResolution);
				}
			}

            if(VerboseFlag & DEBUGINFOS)
            {
			    printf("data allocated and arr structure written\n");
            }

			unsigned char *temp_imgData = (unsigned char*)malloc((perRowBytes * imgObj->vResolution * sizeof(unsigned char)) + 1);

			fseek(imgObj->OriginFilePointer, StripOffsets, SEEK_SET);
			fread(temp_imgData, sizeof(unsigned char), (perRowBytes * imgObj->vResolution) + 1, imgObj->OriginFilePointer);
			
			if(VerboseFlag & DEBUGINFOS)
			{
				printf("temparr initialized and filled");
			}

			for (unsigned long x = 0; x < imgObj->hResolution; x++)
			{
				for (unsigned long y = 0; y < imgObj->vResolution; y++)
				{
					for (unsigned char sample = 0; sample < imgObj->SamplesPerPixel; sample++)
					{
						unsigned int arrptr = (perRowBytes * y) + (((x * BitsPerSample * imgObj->SamplesPerPixel) + (sample * BitsPerSample)) / 8);
						unsigned long LeftJustifiedSubpixelLuminocity = temp_imgData[arrptr] << ((((x * imgObj->SamplesPerPixel * BitsPerSample) + (sample * imgObj->SamplesPerPixel)) % 8) + 24);
						for (unsigned int i = 1; i < (BitsPerSample / 8) + 1; i++)
						{
							LeftJustifiedSubpixelLuminocity |= temp_imgData[arrptr + i] << 32 - ((i + 1) * 8);
						}

						LeftJustifiedSubpixelLuminocity &= (0xffffffff << (32 - BitsPerSample));
						imgObj->RGB_Data[sample][x][y] = (float)LeftJustifiedSubpixelLuminocity / (float)0xffffffff;
					}
				}
				if(VerboseFlag & STATUSBAR)
				{
					printf("\r");
					printf("parsing imgData ");
					printStatusBar((unsigned char)(((float)x / (float)imgObj->hResolution) * 100));
				}
			}
			if(VerboseFlag & STATUSBAR)
			{
				printf("\n");
			}
			free(temp_imgData);
		}
	}

	if ((!(errorflags)) && (ConstructBwFlag)) // convert to B/W and 
	{
        float totalpixelavg;
		imgObj->BW_Data = (float**)malloc(sizeof(float*) * imgObj->hResolution);
		for (unsigned int i = 0; i < imgObj->hResolution; i++)
		{
			imgObj->BW_Data[i] = (float*)malloc(sizeof(float) * imgObj->vResolution);
		}
		
		for (unsigned int x = 0; x < imgObj->hResolution; x++)
		{
			for (unsigned int y = 0; y < imgObj->vResolution; y++)
			{
				float tempval = 0;
				for (unsigned int Sample = 0; Sample < imgObj->SamplesPerPixel; Sample++)
				{
					tempval += powf(imgObj->RGB_Data[Sample][x][y], 2);
				}
				imgObj->BW_Data[x][y] = sqrtf(tempval);
				totalpixelavg += imgObj->BW_Data[x][y] / (imgObj->hResolution * imgObj->vResolution);
			}
		}
	}
	return errorflags;
}

unsigned long IfdReadInt(struct tiff* imgObj, unsigned int IfdTag)
{
    return IFDReadEntry(imgObj->OriginFilePointer, imgObj->IFD.IFDs, imgObj->IFD.Entries, IfdTag);
}

enum errorType closeTiff(struct tiff* imgObj)
{
    enum errorType errorflags;

    free(imgObj->IFD.IFDs);

    for (unsigned int i = 0; i < imgObj-> SamplesPerPixel; i++)
    {
        for (unsigned long j = 0; j < imgObj->hResolution; j++)
        {
            free((void *)imgObj->RGB_Data[i][j]);
        }
        free((void *)imgObj->RGB_Data[i]);
    }
    free((void *)imgObj->RGB_Data);

    for (unsigned int i = 0; i < imgObj->hResolution; i++)
    {
        free(imgObj->BW_Data[i]);
    }
    free(imgObj->BW_Data);

    fclose(imgObj->OriginFilePointer);
}

void printPreview(struct tiff* imgObj, unsigned char prevYres)
{
	unsigned int PixelsPerPixelY = imgObj->vResolution / prevYres;
	unsigned int PixelsPerPixelX = PixelsPerPixelY / 2;
	unsigned int previewXRes = imgObj->hResolution / PixelsPerPixelX;
	unsigned long divBy = PixelsPerPixelX * PixelsPerPixelY * imgObj->SamplesPerPixel;
	float pixelavg = 0;
	float totalpixelavg = 0;

	for (unsigned int x = 0; x < imgObj->hResolution; x++)
	{
		for (unsigned int y = 0; y < imgObj->vResolution; y++)
		{
			float tempval = 0;
			for (unsigned int Sample = 0; Sample < imgObj->SamplesPerPixel; Sample++)
			{
				tempval += powf(imgObj->RGB_Data[Sample][x][y], 2);
			}
			imgObj->BW_Data[x][y] = sqrtf(tempval);
			totalpixelavg += imgObj->BW_Data[x][y] / (imgObj->hResolution * imgObj->vResolution);
		}
	}

	if(VerboseFlag & DEBUGINFOS)
	{
		printf("a preview pixel is composed of %i x %i\n", PixelsPerPixelX, PixelsPerPixelY);
		printf("this results in a image wich is %i x %i\n", prevYres, (unsigned int)previewXRes);
		printf("thus each subpixels has to be divided by %li to summed together result in equal weights\n", divBy);
		printf("the total average luminocity of the image is %f \n", totalpixelavg);
	}
	
	for (unsigned int y = 0; y < prevYres; y++)
	{
		for (unsigned int x = 0; x < previewXRes; x++)
		{
			for (unsigned int orgX = (x * PixelsPerPixelX); (orgX < (PixelsPerPixelX * (x + 1)) && orgX < imgObj->hResolution); orgX++)
			{
				for (unsigned int orgY = (y * PixelsPerPixelY); (orgY < (PixelsPerPixelY * (y + 1)) && orgY < imgObj->vResolution); orgY++)
				{
					pixelavg += imgObj->BW_Data[orgX][orgY], 2;
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


static unsigned char freadchar(FILE *file, unsigned long offset)
{
	fseek(file, offset, SEEK_SET);
	return getc(file);
}

static unsigned int freadint(FILE *file, unsigned long offset)
{
	return (unsigned int)freadchar(file, offset) | ((unsigned int)freadchar(file, offset + 1) << 8);
}

static unsigned long freadlong(FILE *file, unsigned long offset)
{
	return (((unsigned long)freadchar(file, offset)) |
			((unsigned int)freadchar(file, offset + 1) << 8) |
			((unsigned int)freadchar(file, offset + 2) << 16) |
			((unsigned int)freadchar(file, offset + 3) << 24));
}

static unsigned long IFDReadInteger(FILE *file, struct IFD_Entry Entry)
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

static unsigned long IFDReadEntry(FILE *file, struct IFD_Entry *IFD_entries_arr_ptr, unsigned int IFD_Entry_count, unsigned int tag)
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

static unsigned long fillLongInt(FILE *file, unsigned long ZoneOffset, unsigned long bitOffset, unsigned char bitlen)
{
	unsigned long tempval = 0;
	for (unsigned char i = 0; i < 5; i++)
	{
		tempval |= (freadchar(file, ZoneOffset + (bitOffset / 8) + i)) << ((24 - (i * 8)) + (32 - bitlen));
	}
	return tempval | ~(0xffffffff << bitlen);
}

static void printStatusBar(unsigned char input)
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

