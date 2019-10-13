previousWeightDeltas = 0
B = MOMENTUM_CONSTANT
for e in EPOCHS:
    for i in DATASETSIZE:
        output = dataSetInputs[i] * weights
        errors = -(expectedOutput – output)
        localGradients = backPropogation(errors, inputs, weights)
        weightDeltas = (B * previousWeightDeltas) + (learningRate * localGradients * inputs)
        weights = weights + weightDeltas 
		previousWeightDeltas = weightDeltas 




