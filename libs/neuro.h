#include "stdlib.h"
#include "stdio.h"
#define SIGMOD 1
#define NODES 3
#define LAYERS 3
#define WEIGHTDELTASTEP 0.1
#define BIASDELTASTEP 0.1

//float testweighting[LAYERS][NODES][(NODES + 1)] /*[layer][nodes][per input before weighting, n+1 bias]*/ = {
//    {{-10, 0, 0, 0},
//     {0, 0, 0, 0},
//     {0, 0, 0, 0}},
//    {{-10, 0, 0, 0},
//     {0, 0, 0, 0},
//     {0, 0, 0, 0}},
//    {{-10, 0, 0, 0},
//     {0, 0, 0, 0},
//     {0, 0, 0, 0}}};
//float NeuronActivations[LAYERS][NODES];
//float inputvalues[NODES] = {0.5, 0.5, 0.5};
//float supposedValue[NODES] = {1, 0, 0};
//float testInput[NODES] = {1, 0, 0};
//float value = 1;
//uint32_t runtimeUs = 0;


// new funcs
enum errorType /*: unsigned int*/
{
    Success = 0x00,
    FileAccessError = 0x01,
    FileIsNotTiffError = 0x02,
    FileIsBigEndianError = 0x04,
    UnsupportedEncodingError = 0x08,
    MultiStripImageError = 0x10
};

enum layerType
{
    inputLayer = 0,
    hiddenLayer = 1,
    outputLayer = -1
};

struct inputLayer
{
    unsigned int nodes;
    float * inputActivations;
};

struct hiddenLayer
{
    unsigned int nodes;
    float ** weights;
    float * biases;
    float (*delinfunc) (float);
};

struct neuronalNetwork
{
    struct inputLayer inputLayer;
    struct hiddenLayer * hiddenLayers;
    unsigned int hiddenLayerCount;
    FILE * NNWBFile;
};

typedef enum errorType error;

error initInputLayer(struct inputLayer * layer, unsigned int size);
error initHiddenLayer(struct hiddenLayer * layer_n1, void * layer_n0, unsigned int neurons, enum layerType layer_N0_type);
error initNeuronalNetwork(struct neuronalNetwork * Network, unsigned int perLayerNeurons [], unsigned int layers);
void freeInputLayer(struct inputLayer * layer);
void freeHiddenLayer(struct hiddenLayer * layer);
void freeNeuronalNetwork(struct neuronalNetwork * Network);
error initNeuronalNetworFromFile(struct neuronalNetwork * networkPtr);
error saveNeuronalNetworToFile(struct neuronalNetwork * networkPtr);


// until here


float sigmoid(float inputvalue);
void calcArr(float inputarr[NODES], float weightings[LAYERS][NODES][(NODES + 1)]);
void calcArrSaveActivations(float inputarr[NODES], float NeuronActivations[LAYERS + 1][NODES], float weightings[LAYERS][NODES][(NODES + 1)]);
float calcNode(float prevLayerActivation[NODES], float weighting[(NODES + 1)]);
float cost(float supposedValue[NODES], float result[NODES]);
float average(float values[], unsigned char numberOfValues);
void train(float weightsNBiases[LAYERS][NODES][NODES + 1], float supposedValues[NODES], float inputvalues[NODES]);
void backpropagation(float weightsNBiases[LAYERS][NODES][NODES + 1], float supposedValues[NODES], float inputvalues[NODES]);