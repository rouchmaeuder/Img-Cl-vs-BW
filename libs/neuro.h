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

float sigmoid(float inputvalue);
void calcArr(float inputarr[NODES], float weightings[LAYERS][NODES][(NODES + 1)]);
void calcArr(float inputarr[NODES], float NeuronActivations[LAYERS + 1][NODES], float weightings[LAYERS][NODES][(NODES + 1)]);
float calcNode(float prevLayerActivation[NODES], float weighting[(NODES + 1)]);
float cost(float supposedValue[NODES], float result[NODES]);
float average(float values[], unsigned char numberOfValues);
void train(float weightsNBiases[LAYERS][NODES][NODES + 1], float supposedValues[NODES], float inputvalues[NODES]);
void backpropagation(float weightsNBiases[LAYERS][NODES][NODES + 1], float supposedValues[NODES], float inputvalues[NODES]);