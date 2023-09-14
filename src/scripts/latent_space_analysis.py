import torch
import numpy as np
import os
import argparse
import h5py
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.utils.data as data

from models import model_dict
from visualization import load_model
from torchvision import transforms
from utils import OnePixelAttack
from dataset import Galaxy10DECalsTest

def load_data(original_data_filname, perturbed_data_filename):
	perturbed_dataset = OnePixelAttack(perturbed_data_filename)

	transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            transforms.Resize(255)
        ])

	custom_idxs = np.load("../../../data/all_correct_idxs.npy", allow_pickle=True)
	original_images = Galaxy10DECalsTest(original_data_filname, transform, custom_idxs=custom_idxs)

	perturbed_images = perturbed_dataset.get_image()
	num = len(perturbed_dataset)
	perturbed_images = perturbed_images.reshape((num,3 , perturbed_images.shape[1], perturbed_images.shape[2]))
	original_images = data.Subset(original_images, perturbed_dataset.get_idx())
	return original_images, perturbed_images

def get_latent_space_represenatation(model, images):
	
	model = model.to(device)
	images = images.to(device)
	latent_space_representation = model(images)
	return latent_space_representation

	
	
if __name__ == '__main__':
	print("Load data!")
	parser = argparse.ArgumentParser(description = 'Latent Space Analysis')
	parser.add_argument('--data_path', type=str, metavar='original_data_filname', required=True,
						help='Location of original dataset')
	parser.add_argument('--perturbed_data', type=str, metavar='perturbed_data_filename', required=True,
						help='Location of perturbed dataset')
	parser.add_argument('--model_name', type=str, metavar='model_path', required=True,
						help='Name of model')
	parser.add_argument('--model_path', type=str, metavar='model_path', required=True,
						help='Location of model')

	args = parser.parse_args()

	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	original_images, perturbed_images = load_data(args.data_path, args.perturbed_data)

	if any(char.isdigit() for char in args.model_name):
		is_equivarient = True
	
	feature_model = load_model(args.model_path, args.model_name, is_equivarient)

	original_latent_space_representation = get_latent_space_represenatation(feature_model, original_images)

	print("Original Latent Space Representation Shape: ", original_latent_space_representation.shape)
