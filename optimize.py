# Import comet_ml in the top of your file
from comet_ml import Experiment, Optimizer
from train import train
# Create an experiment


def main():
	optimizer = Optimizer("vwvQEjQJwXYzVinuhRLVGYkLy")

	# Declare your hyper-parameters in the PCS format
	params = """
	h_dim integer [2, 2048] [256]
	z_dim integer [2, 2048] [256]
	batch_size integer [1, 128] [64]
	n_layers integer [1,3] [2]
	initial_lr real [.00001, .01] [.001]
	weight_decay real [.000005, .5] [.0005]
	scheduling_start real [.3, .8] [.5]
	scheduling_end real [0.0, 0.3] [.10]
	"""

	optimizer.set_params(params)

	# get_suggestion will raise when no new suggestion is available
	while True:
		# Get a suggestion
		suggestion = optimizer.get_suggestion()

		# Create a new experiment associated with the Optimizer
		# import comet_ml in the top of your file
		experiment = Experiment(api_key="vwvQEjQJwXYzVinuhRLVGYkLy",
		project_name="trafficrnn", workspace="zeiberg-d")

		# Test the model
		train_loss, val_loss, save_dir = train(suggestion)

		# Report the score back
		suggestion.report_score("train_loss", train_loss)
		suggestion.report_score("validation_loss",val_loss)
		suggestion.report_score("save_dir", save_dir)

"""
def testVals(x, y):
	return  ((5-x)**2 + (.5-y)**2)

def testComet():
	optimizer = Optimizer("vwvQEjQJwXYzVinuhRLVGYkLy")

	# Declare your hyper-parameters in the PCS format
	params =
	#x integer [1, 10] [3]
	#y real [0, 2] [1.7]

	optimizer.set_params(params)

	# get_suggestion will raise when no new suggestion is available
	while True:
	  # Get a suggestion
	  suggestion = optimizer.get_suggestion()

	  # Create a new experiment associated with the Optimizer
	  experiment = Experiment(api_key="vwvQEjQJwXYzVinuhRLVGYkLy", project_name="trafficrnn", workspace="zeiberg-d")

	  # Test the model
	  loss = testVals(suggestion["x"], suggestion["y"])

	  # Report the score back
	  suggestion.report_score("loss",loss)
"""

if __name__ == '__main__':
	main()
