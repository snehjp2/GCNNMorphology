import torch
import numpy as np
import os
import argparse
import h5py
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
from scipy.spatial import distance
import pandas as pd
import seaborn as sns
from torch import nn
sns.set()
import matplotlib.pyplot as plt
import umap

from models import model_dict
from visualization import load_model
from torchvision import transforms
from utils import OnePixelAttack
from dataset import Galaxy10DECalsTest

def load_perturbed_data(original_data_filname, perturbed_data_filename):
	'''
	Load the perturbed data and it's original data from the dataset
	'''
	perturbed_dataset = OnePixelAttack(perturbed_data_filename)

	transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            transforms.Resize(255)
        ])

	custom_idxs = np.load("/path/to/all_correct_idxs.npy", allow_pickle=True)
	original_images = Galaxy10DECalsTest(original_data_filname, transform, custom_idxs=custom_idxs)

	perturbed_images = perturbed_dataset.get_image()
	num = len(perturbed_dataset)
	perturbed_images = perturbed_images.reshape((num,3 , perturbed_images.shape[1], perturbed_images.shape[2]))
	original_images = data.Subset(original_images, perturbed_dataset.get_img_idx())
	perturbed_images = torch.from_numpy(perturbed_images)
	return original_images, perturbed_images

def load_noisy_data(original_data_filname, noisy_data_25, noisy_data_50):
	'''
	Load the noisy data and it's original data from the dataset
	'''
	transform = transforms.Compose([
		transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            transforms.Resize(255)
        ])

	custom_idxs = np.load("/path/to/all_correct_idxs.npy", allow_pickle=True)
	original_images = Galaxy10DECalsTest(original_data_filname, transform, custom_idxs=custom_idxs)
	noisy_images_25 = Galaxy10DECalsTest(noisy_data_25, transform, custom_idxs=custom_idxs)
	noisy_images_50 = Galaxy10DECalsTest(noisy_data_50, transform, custom_idxs=custom_idxs)

	return original_images, noisy_images_25, noisy_images_50

def get_latent_space_represenatation(model, images, label):
	'''
	Get the latent space representation for the images

	'''
	model = model.to(device)
	images = images.to(device)
	latent_space_representation, output = model(images)
	outputs = torch.argmax(output, dim=-1).cpu().numpy()
	outputs = np.array(outputs)

	label = label.cpu().detach().numpy()
	label = np.array(label)
	misclassified_indices = np.where(label != outputs)[0]

	return latent_space_representation, misclassified_indices

	
	
if __name__ == '__main__':

	parser = argparse.ArgumentParser(description = 'Latent Space Analysis')
	parser.add_argument('--data_path', type=str, metavar='original_data_filname', required=True,
						help='Location of original dataset')
	parser.add_argument('--noisy_data_25', type=str, metavar='noisy_data_dir', required=True,
						help='Location of 25% noisy dataset')
	parser.add_argument('--noisy_data_50', type=str, metavar='noisy_data_dir', required=True,
						help='Location of 50% noisy dataset')
	parser.add_argument('--perturbed_data_dir', type=str, metavar='perturbed_data_dir', required=True,
						help='Location of perturbed dataset')
	parser.add_argument('--model_dir', type=str, metavar='model_dir', required=True,
						help='Location of model directory')

	args = parser.parse_args()

	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')



	mean_noisy_25 = {}
	mean_noisy_50 = {}

	for file_name in os.listdir(args.model_dir):
		file_path = os.path.join(args.model_dir, file_name)
		if file_name.endswith('.pt') and os.path.isfile(file_path):
			print(f'Loading {file_name}...')
			model_name = os.path.splitext(file_name)[0]

			is_equivarient = False
			if any(char.isdigit() for char in model_name):
				is_equivarient = True
    
			## do inference with full model -- if output == true, pass

			feature_model = load_model(file_path, model_name, is_equivarient)	
			feature_model = nn.DataParallel(feature_model)

			original_images, noisy_images_25, noisy_images_50 = load_noisy_data(args.data_path, args.noisy_data_25, args.noisy_data_50)

			original_images = DataLoader(original_images, batch_size=128, shuffle=False)
			noisy_images_25 = DataLoader(noisy_images_25, batch_size=128, shuffle=False)
			noisy_images_50 = DataLoader(noisy_images_50, batch_size=128, shuffle=False)

			original_latent_space_representation = []
			noisy_25_latent_space_representation = []
			noisy_50_latent_space_representation = []
   
			for img, label, _, _ in original_images:
				features, idx = get_latent_space_represenatation(feature_model, img, label)
				original_latent_space_representation.append(features.cpu().detach().numpy())

			original_latent_space_representation = np.concatenate(original_latent_space_representation, axis=0)
			
			current_start_idx = 0
			batch_size = 128  # Since your DataLoader uses a batch size of 128

			noisy_25_misclassified_idx_global = []  # List to hold the adjusted indices

			# Loop through the DataLoader using enumerate
			for batch_idx, (img, label, _, _) in enumerate(noisy_images_25):
				features, misclassified_idx = get_latent_space_represenatation(feature_model, img, label)
				noisy_25_latent_space_representation.append(features.cpu().detach().numpy())
				
				# Adjust the misclassified indices
				misclassified_idx_global = misclassified_idx + current_start_idx
				noisy_25_misclassified_idx_global.append(misclassified_idx_global)

				# Update the current starting index for the next iteration
				current_start_idx += batch_size

			# Concatenate the adjusted indices
			noisy_25_misclassfied_idx_global = np.concatenate(noisy_25_misclassified_idx_global, axis=0)
			noisy_25_latent_space_representation = np.concatenate(noisy_25_latent_space_representation, axis=0)

			current_start_idx = 0
			noisy_50_misclassified_idx_global = []
			for batch_idx, (img, label, _, _) in enumerate(noisy_images_50):
				features, misclassified_idx = get_latent_space_represenatation(feature_model, img, label)
				noisy_50_latent_space_representation.append(features.cpu().detach().numpy())
				
				misclassified_idx_global = misclassified_idx + current_start_idx
				noisy_50_misclassified_idx_global.append(misclassified_idx_global)
    
				current_start_idx += batch_size
			
			noisy_50_latent_space_representation = np.concatenate(noisy_50_latent_space_representation, axis=0)
			noisy_50_misclassfied_idx_global = np.concatenate(noisy_50_misclassified_idx_global, axis=0)

			x = original_latent_space_representation[noisy_25_misclassfied_idx_global]
			y = noisy_25_latent_space_representation[noisy_25_misclassfied_idx_global]

			mean_noisy_25[model_name] = np.mean(np.linalg.norm(x - y, axis=1))

			x = original_latent_space_representation[noisy_50_misclassfied_idx_global]
			y = noisy_50_latent_space_representation[noisy_50_misclassfied_idx_global]	
			mean_noisy_50[model_name] = np.mean(np.linalg.norm(x - y, axis=1))

			print(f"Mean distance between original and 25% noisy images in the latent space for {model_name}: ", mean_noisy_25[model_name])
			print(f"Mean distance between original and 50% noisy images in the latent space for {model_name}: ", mean_noisy_50[model_name])
			

	##save dict as np array
	

	mean_noisy_25_key = list(mean_noisy_25.keys())
	mean_noisy_25_value = list(mean_noisy_25.values())
	combined_array = np.vstack((mean_noisy_25_key, mean_noisy_25_value)).T
	print(combined_array)
	np.save('/path/to/mean_noisy_25.npy', combined_array)

	mean_noisy_50_key = list(mean_noisy_50.keys())
	mean_noisy_50_value = list(mean_noisy_50.values())
	combined_array = np.vstack((mean_noisy_50_key, mean_noisy_50_value)).T
	print(combined_array)
	np.save('path/to/mean_noisy_50.npy', combined_array)

