#include "neuro.h"
#include "stdlib.h"
#define SIGMOID
#define PREDEFFUNCCOUNT 3

static unsigned char freadchar(FILE *file, unsigned long offset);
static unsigned int freadint(FILE *file, unsigned long offset);
static unsigned long freadlong(FILE *file, unsigned long offset);


error initInputLayer(struct inputLayer * layer, unsigned int size);
error initHiddenLayer(struct hiddenLayer * layer_n1, void * layer_n0, unsigned int neurons, enum layerType layer_N0_type);
error initNeuronalNetwork(struct neuronalNetwork * Network, unsigned int perLayerNeurons [], unsigned int layers);
void freeInputLayer(struct inputLayer * layer);
void freeHiddenLayer(struct hiddenLayer * layer);
void freeNeuronalNetwork(struct neuronalNetwork * Network);
error initNeuronalNetworFromFile(struct neuronalNetwork * networkPtr)
{
    __uint32_t index = 0;
    networkPtr->hiddenLayerCount = freadlong(networkPtr->NNWBFile, index * 4);
    index++;
    networkPtr->inputLayer.nodes = freadlong(networkPtr->NNWBFile, index * 4);
    index++;
    networkPtr->hiddenLayers = malloc(sizeof(struct hiddenLayer) * networkPtr->hiddenLayerCount);
    for (__uint16_t i = 0; i < networkPtr->hiddenLayerCount; i++)
    {
        networkPtr->hiddenLayers[i].nodes = freadlong(networkPtr->NNWBFile, index * 4);
        index++;
        networkPtr->hiddenLayers[i].weights = malloc(sizeof(float*) * networkPtr->hiddenLayers[i].nodes);
        for (__uint16_t NodeN1 = 0; NodeN1 < networkPtr->hiddenLayers[i].nodes; NodeN1++) // initializing weights
        {
            __uint16_t prevLayerNodes;
            if (i)
            {
                prevLayerNodes = networkPtr->hiddenLayers->nodes;
            }
            else
            {
                prevLayerNodes = networkPtr->inputLayer.nodes;
            }

            networkPtr->hiddenLayers[i].weights[NodeN1] = malloc(sizeof(float) * prevLayerNodes);
            for (__uint16_t NodeN0 = 0; NodeN0 < prevLayerNodes; NodeN0++)
            {
                __uint32_t conversionValue = freadlong(networkPtr->NNWBFile, index * 4);
                index++;
                networkPtr->hiddenLayers[i].weights[NodeN1][NodeN0] = *((float*)(&conversionValue));
            }
        }

        networkPtr->hiddenLayers[i].biases = malloc(sizeof(float) * networkPtr->hiddenLayers[i].nodes);
        for (__uint16_t NodeN1 = 0; NodeN1 < networkPtr->hiddenLayers[i].nodes; NodeN1++) // initializing biasies
        {
            __uint32_t conversionValue = freadlong(networkPtr->NNWBFile, index * 4);
            index++;
            networkPtr->hiddenLayers[i].biases[NodeN1] = *((float*)(&conversionValue));
        }
        __uint32_t tempFuncIdentifier = freadlong(networkPtr->NNWBFile, index * 4);
        index++;
        if (tempFuncIdentifier >= PREDEFFUNCCOUNT)
        {
            networkPtr->hiddenLayers[i].delinfunc = delinFuncLut[tempFuncIdentifier];
        }
        else
        {
            networkPtr->hiddenLayers[i].delinfunc = NULL;
        }
    }
    
    return 0;
}
error saveNeuronalNetworToFile(struct neuronalNetwork * networkPtr) // DANGER!! overwrites the NNWB file!
{
    __uint32_t index = 0;
    char * filepath;
    __uint32_t filesize = 0;
    fgetpos(networkPtr->NNWBFile, filepath);
    fclose(networkPtr->NNWBFile);
    fopen(filepath, "wb");

    filesize += 2;
    __uint32_t * binStream = malloc(2 * sizeof(__uint32_t));

    binStream[index] = networkPtr->hiddenLayerCount;
    index++;
    binStream[index] = networkPtr->inputLayer.nodes;
    index++;
  //  networkPtr->hiddenLayers = malloc(sizeof(struct hiddenLayer) * networkPtr->hiddenLayerCount);

    filesize += 
    for (__uint16_t i = 0; i < networkPtr->hiddenLayerCount; i++)
    {
        networkPtr->hiddenLayers[i].nodes = freadlong(networkPtr->NNWBFile, index * 4);
        index++;
        networkPtr->hiddenLayers[i].weights = malloc(sizeof(float*) * networkPtr->hiddenLayers[i].nodes);
        for (__uint16_t NodeN1 = 0; NodeN1 < networkPtr->hiddenLayers[i].nodes; NodeN1++) // initializing weights
        {
            __uint16_t prevLayerNodes;
            if (i)
            {
                prevLayerNodes = networkPtr->hiddenLayers->nodes;
            }
            else
            {
                prevLayerNodes = networkPtr->inputLayer.nodes;
            }

            networkPtr->hiddenLayers[i].weights[NodeN1] = malloc(sizeof(float) * prevLayerNodes);
            for (__uint16_t NodeN0 = 0; NodeN0 < prevLayerNodes; NodeN0++)
            {
                __uint32_t conversionValue = freadlong(networkPtr->NNWBFile, index * 4);
                index++;
                networkPtr->hiddenLayers[i].weights[NodeN1][NodeN0] = *((float*)(&conversionValue));
            }
        }

        networkPtr->hiddenLayers[i].biases = malloc(sizeof(float) * networkPtr->hiddenLayers[i].nodes);
        for (__uint16_t NodeN1 = 0; NodeN1 < networkPtr->hiddenLayers[i].nodes; NodeN1++) // initializing biasies
        {
            __uint32_t conversionValue = freadlong(networkPtr->NNWBFile, index * 4);
            index++;
            networkPtr->hiddenLayers[i].biases[NodeN1] = *((float*)(&conversionValue));
        }
        __uint32_t tempFuncIdentifier = freadlong(networkPtr->NNWBFile, index * 4);
        index++;
        if (tempFuncIdentifier >= PREDEFFUNCCOUNT)
        {
            networkPtr->hiddenLayers[i].delinfunc = delinFuncLut[tempFuncIdentifier];
        }
        else
        {
            networkPtr->hiddenLayers[i].delinfunc = NULL;
        }
    }
    
    return 0;
}


// old funcs

void calcArr(float inputarr[NODES], float weightings[LAYERS][NODES][(NODES + 1)])
{
    for (unsigned char i = 0; i < LAYERS; i++) // loop through layers
    {
        for (unsigned char j = 0; j < NODES; j++) // loo p through nodes
        {
            inputarr[j] = calcNode(inputarr, weightings[i][j]);
        }
    }
}

void calcArrSaveActivations(float inputarr[NODES], float NeuronActivations[LAYERS + 1][NODES], float weightings[LAYERS][NODES][(NODES + 1)])
{
    for (unsigned char i = 0; i < LAYERS; i++) // clearing the neuron activations arr and inserting inputvalues
    {
        for (unsigned char j = 0; j < NODES; j++)
        {
            if (!(i))
            {
                NeuronActivations[i][j] = 0;
            }
            else
            {
                NeuronActivations[i][j] = inputarr[j];
            }
        }
    }

    for (unsigned char i = 0; i < LAYERS; i++) // loop through layers
    {
        for (unsigned char j = 0; j < NODES; j++) // loop through nodes
        {
            NeuronActivations[i + 1][j] = calcNode(NeuronActivations[i], weightings[i][j]);
        }
    }
}

float calcNode(float prevLayerActivation[NODES], float weighting[NODES + 1])
{
    unsigned char i;
    float activation = 0;
    for (i = 0; i < NODES; i++)
    {
        activation += prevLayerActivation[i] * weighting[i]; // weighted sum all previous nodes
    }
    activation += weighting[NODES]; // add bias (danger must be only nodes beacause amount != designator)
#ifdef SIGMOID
    activation = sigmoid(activation);
#else
    activation = sigmoid(activation);
#endif
    return activation;
}

float sigmoid(float inputvalue)
{
    return (1 / (1 + pow(3, (0 - inputvalue))));
}

float cost(float supposedValue[NODES], float result[NODES])
{
    float cost = 0;
    for (unsigned char i = 0; i < NODES; i++)
    {
        cost += pow((result[i] - supposedValue[i]), 2);
    }
    return cost;
}

float average(float values[], unsigned char numberOfValues)
{
    float average = 0;
    for (unsigned char i = 0; i < numberOfValues; i++)
    {
        average += values[i];
    }
    return (average / numberOfValues);
}

void train(float weightsNBiases[LAYERS][NODES][NODES + 1], float supposedValues[NODES], float inputvalues[NODES])
{
    float beforecost = 0;
    float aftercost = 0;
    float vector[LAYERS][NODES][NODES + 1];
    float testarr[NODES];
    for (unsigned char x = 0; x < NODES; x++)
    {
        testarr[x] = inputvalues[x];
    }
    calcArr(testarr, weightsNBiases);
    beforecost = cost(supposedValues, testarr);

    for (unsigned char i = 0; i < LAYERS; i++) // init vector array to zero
    {
        for (unsigned char j = 0; j < NODES; j++)
        {
            for (unsigned char y = 0; y < (NODES + 1); y++)
            {
                vector[i][j][y] = 0;
            }
        }
    }

    for (unsigned char i = 0; i < LAYERS; i++) // modify each weight and bias and test the effect
    {
        for (unsigned char j = 0; j < NODES; j++)
        {
            for (unsigned char y = 0; y < (NODES + 1); y++)
            {
                for (unsigned char x = 0; x < NODES; x++) // copy inputvalues into working array
                {
                    testarr[x] = inputvalues[x];
                }
                if (y < NODES)
                {
                    weightsNBiases[i][j][y] += WEIGHTDELTASTEP;
                }
                else
                {
                    weightsNBiases[i][j][y] += BIASDELTASTEP;
                }
                calcArr(testarr, weightsNBiases);
                aftercost = cost(supposedValues, testarr);
                vector[i][j][y] = beforecost - aftercost;
                if (y < NODES)
                {
                    weightsNBiases[i][j][y] -= WEIGHTDELTASTEP;
                }
                else
                {
                    weightsNBiases[i][j][y] -= BIASDELTASTEP;
                }
            }
        }
    }
    for (unsigned char i = 0; i < LAYERS; i++) // init vector array to zero
    {
        for (unsigned char j = 0; j < NODES; j++)
        {
            for (unsigned char y = 0; y < (NODES + 1); y++)
            {
                weightsNBiases[i][j][y] += (0.1 * vector[i][j][y]);
            }
        }
    }
}

void backpropagation(float weightsNBiases[LAYERS][NODES][NODES + 1], float supposedValues[NODES], float inputvalues[NODES])
{
    float vector[LAYERS][NODES][NODES + 1];
    float Activations[LAYERS + 1][NODES];
    float OldActivationChanges[NODES];
    float ActivationChanges[NODES];
    float supposedValueCopy[NODES];

    for (unsigned char i = 0; i < NODES; i++)
    {
        supposedValueCopy[i] = supposedValues[i];
    }
    

    calcArr(inputvalues, Activations, weightsNBiases);

    for (unsigned char i = (LAYERS); i != 0; i--) // loop through layers, 1 being the last layer processed, corresponding to the first hidden layer and its weights n biases
    {
        for (unsigned char j = 0; j < NODES; j++) // loop through nodes and calculate optimum activation changes, j being the node from 0 to NODES
        {
            ActivationChanges[j] = supposedValueCopy[j] - Activations[i][j];
            supposedValueCopy[j] = 0;
        }
        for (unsigned char j = 0; j < NODES; j++) // loop through nodes, j being the nodes
        {
            for (unsigned char y = 0; y < NODES; y++) // loop through wheights for node
            {
                vector[i-1][j][y] = ActivationChanges[j] * Activations[i - 1][y]; // the optimum weight change is the activation in the previous layer corresponding to that weight(Activation[i-1])

                supposedValueCopy[j] += ActivationChanges[y] * weightsNBiases[i-1][j][y]; // calculate next supposed values
            }
            vector[i-1][j][NODES] = ActivationChanges[j]; // set bias vector component
        }
    }


    for (unsigned char i = 0; i < LAYERS; i++)
    {
        for (unsigned char j = 0; j < NODES; j++)
        {
            for (unsigned char y = 0; y < NODES + 1; y++)
            {
                weightsNBiases[i][j][y] += (vector[i][j][y] * WEIGHTDELTASTEP);
            }
        }
    }
}

void setWeightsNBiasarrToZero(float weightsNBiases[LAYERS][NODES][NODES + 1])
{
    for (unsigned char i = 0; i < LAYERS; i++) // init vector array to zero
    {
        for (unsigned char j = 0; j < NODES; j++)
        {
            for (unsigned char y = 0; y < (NODES + 1); y++)
            {
                weightsNBiases[i][j][y] = 0;
            }
        }
    }
}

// file funcs

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