void									printStatusBar(unsigned char input);																					   // takes a unsigned char from 0 to 100 as a percentage
__device__ static inline signed long	limit(signed long input, signed long lower, signed long upper);
__global__ void							ParalellTotalContrastParallelismFunction (float **InputData, float **outputData, unsigned int frame, unsigned long index, unsigned int hResolution, unsigned int vResolution); // supposed to be run in threads
float									ParalellTotalContrast(float **image, float radius);
