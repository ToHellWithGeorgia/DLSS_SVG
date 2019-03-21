import argparse

def get_train_args():
	parser = argparse.ArgumentParser('Train a VAE model.')

	add_train_test_args(parser)

	parser.add_argument('--train_dataset_dir',
						type=str,
						default='dataset/train/',
						help='Directory of the training dataset')
	parser.add_argument('--dev_dataset_dir',
						type=str,
						default='dataset/dev/',
						help='Directory of the dev dataset')
	parser.add_argument('--eval_steps',
						type=int,
						default=500,
						help='Number of steps between evaluation.')
	parser.add_argument('--lr',
						type=float,
						default=1e-4,
						help='Learning rate.')
	parser.add_argument('--max_grad_norm',
						type=int,
						default=5,
						help='Maximum gradient norm')
	parser.add_argument('--num_epochs',
						type=int,
						default=500,
						help='Number of epochs to train.')
	parser.add_argument('--seed',
						type=int,
						default=248,
						help='random seed for all rng.')

	args, _ = parser.parse_known_args()
	return args

def get_test_args():
	parser = argparse.ArgumentParser('Test a VAE model.')

	add_train_test_args(parser)

	parser.add_argument('--test_dataset_dir',
						type=str,
						default='dataset/test/',
						help='Directory of the test dataset')

	args, _ = parser.parse_known_args()
	return args


def add_train_test_args(parser):
	parser.add_argument('--name',
						'-n',
						type=str,
						default='test',
						# required=True,
						help='Name to identify training or test run.')
	parser.add_argument('--save_dir',
						type=str,
						default='./save/',
						help='Base directory for saving information')
	parser.add_argument('--batch_size',
						type=int,
						default=8,
						help='batch size of training or testing')