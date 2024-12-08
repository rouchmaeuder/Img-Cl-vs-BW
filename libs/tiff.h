#include "stdlib.h"
#include "stdio.h"
#include "math.h"

#define BYTE_TYPE 1
#define ASCII_TYPE 2
#define INT_TYPE 3
#define LONG_TYPE 4
#define RATIO_TYPE 5

#define RED 0
#define GREEN 1
#define BLUE 2

#define ERRORS 0x01
#define STATS 0x02
#define STATUSBAR 0x04
#define DEBUGINFOS 0x80

enum errorType /*: unsigned int*/
{
    Success = 0x00,
    FileAccessError = 0x01,
    FileIsNotTiffError = 0x02,
    FileIsBigEndianError = 0x04,
    UnsupportedEncodingError = 0x08,
    MultiStripImageError = 0x10
};
/*
bit  |error
0x01 |file could not be opened
0x02 |file does not have necessary 42 at ptr+2
0x04 |file is in big endian
0x08 |unsupported encoding type
0x10 |above 1 img strip
*/

enum verbosityLevel{
    PrintNone = 0x00,
    PrintErrors = 0x01,
    PrintStats = 0x02,
    PrintErrorsAndStats = 0x03,
    PrintStatusBar = 0x04,
    PrintDebugStatsAndErrors = 0x87
};
/*
bit  |prints
0x01 |errors
0x02 |stats
0x04 |statusBar
0x80 |debuginfos
*/

extern enum verbosityLevel VerboseFlag;

enum IFDDataType
{
    BYTE = 1,
    CHAR = 2,
    SHORT = 3,
    LONG = 4,
    RATIONAL = 5
};

struct tiff
{
    unsigned long hResolution;
    unsigned long vResolution;
    unsigned char SamplesPerPixel;
    struct IFD
    {
        unsigned int Entries;
        enum IFDDataType *Type;
        struct IFD_Entry *IFDs;
    } IFD;

    FILE* OriginFilePointer;
    float *** RGB_Data;
    unsigned char BW_DataIsAvailable;
    float ** BW_Data;
};

struct IFD_Entry
{
    unsigned int tag;
    unsigned int type;
    unsigned long data_arr_len;
    unsigned long data_offset;
};

enum errorType openTiff(struct tiff* imgObj, unsigned char ConstructBwFlag, char filePath []);

unsigned long IfdReadInt(struct tiff* imgObj, unsigned int IfdTag);

void printPreview(struct tiff* imgObj, unsigned char prevYres);

enum errorType closeTiff(struct tiff* imgObj);