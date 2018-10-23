import numpy as np
import cProfile
import model.Data as DF
import model.EncoderRNN as ERF
import model.AttnDecoder as ADRF
import utils
import pstats

modelParams = {
	'n_layers': 2,
	'hidden_state_size': 64
}

def main(trainParams=None):
        print("loading data")
	# Get data objects
        TrafficDataTrainObj = DF.Data("./data/train.npz")
        TrafficDataValObj = DF.Data("./data/val.npz")

        # Get data constants
        sequence_length = TrafficDataValObj.getSequenceLength()
        num_features = TrafficDataTrainObj.getNumFeatures()

        # Define Model
        encoder = ERF.EncoderRNN(sequence_length, num_features,
                                 modelParams['hidden_state_size'], n_layers=modelParams['n_layers'])

        decoder = ADRF.AttnDecoderRNN(modelParams['hidden_state_size'], num_features,
                                     n_layers=modelParams['n_layers'], method="general")

        modelDescription = "Encoder Decoder RNN w/ Attn - hidden state size: {}, layers: {}".format(modelParams['hidden_state_size'], modelParams['n_layers'])
        if trainParams and trainParams["n_epochs"] and trainParams["batchesPerEpoch"]:
                trainLosses, valLosses = utils.train(TrafficDataTrainObj, TrafficDataValObj, encoder, decoder,
                        modelDescription, n_epochs=trainParams["n_epochs"], batchesPerEpoch=trainParams["batchesPerEpoch"])
        else:
                trainLosses, valLosses = utils.train(TrafficDataTrainObj, TrafficDataValObj, encoder, decoder, modelDescription)

if __name__ == '__main__':
        cProfile.run("main(trainParams={'n_epochs':2, 'batchesPerEpoch':2})", "restats")
        p = pstats.Stats('restats')
        p.sort_stats("tottime").print_stats()
