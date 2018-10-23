import cProfile
import model.Data as DF
import model.EncoderRNN as ERF
import model.AttnDecoder as ADRF
import utils

modelParams = {
	'n_layers': 2,
	'hidden_state_size': 64
}

def main(modelParams=None):
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
	if trainParams and modelParams["n_epochs"] and modelParams["batchesPerEpoch"]:
		trainLosses, valLosses = utils.train(TrafficDataTrainObj, TrafficDataValObj, encoder, decoder,
			modelDescription, n_epochs=modelParams["n_epochs"], batchesPerEpoch=modelParams["batchesPerEpoch"])
	else:
		trainLosses, valLosses = utils.train(TrafficDataTrainObj, TrafficDataValObj, encoder, decoder, modelDescription)

if __name__ == '__main__':
	cProfile.run("main(modelParams={n_epochs=2, batchesPerEpoch=20})")
	main()
