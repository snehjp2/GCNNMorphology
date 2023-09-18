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
sns.set()
import matplotlib.pyplot as plt
import umap

from models import model_dict
from visualization import load_model
from torchvision import transforms
from utils import OnePixelAttack
from dataset import Galaxy10DECalsTest

def load_perturbed_data(original_data_filname, perturbed_data_filename):
	perturbed_dataset = OnePixelAttack(perturbed_data_filename)

	transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            transforms.Resize(255)
        ])

	custom_idxs = np.load("/work/GDL/all_correct_idxs.npy", allow_pickle=True)
	original_images = Galaxy10DECalsTest(original_data_filname, transform, custom_idxs=custom_idxs)

	perturbed_images = perturbed_dataset.get_image()
	num = len(perturbed_dataset)
	perturbed_images = perturbed_images.reshape((num,3 , perturbed_images.shape[1], perturbed_images.shape[2]))
	original_images = data.Subset(original_images, perturbed_dataset.get_img_idx())
	perturbed_images = torch.from_numpy(perturbed_images)
	return original_images, perturbed_images

def load_noisy_data(original_data_filname, noisy_data_25, noisy_data_50):
	transforms = transforms.Compose([
		transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            transforms.Resize(255)
        ])

	custom_idxs = np.load("/work/GDL/all_correct_idxs.npy", allow_pickle=True)
	original_images = Galaxy10DECalsTest(original_data_filname, transforms, custom_idxs=custom_idxs)
	noisy_images_25 = Galaxy10DECalsTest(noisy_data_25, transforms, custom_idxs=custom_idxs)
	noisy_images_50 = Galaxy10DECalsTest(noisy_data_50, transforms, custom_idxs=custom_idxs)

	return original_images, noisy_images_25, noisy_images_50

def get_latent_space_represenatation(model, images):
	
	model = model.to(device)
	images = images.to(device)
	latent_space_representation = model(images)
	return latent_space_representation

	
	
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
	# parser.add_argument('--model_name', type=str, metavar='model_path', required=True,
	# 					help='Name of model')
	# parser.add_argument('--model_path', type=str, metavar='model_path', required=True,
	# 					help='Location of model')

	args = parser.parse_args()

	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	
	mean_perturbed = {}


	for file_name in os.listdir(args.model_dir):
		file_path = os.path.join(args.model_dir, file_name)
		if file_name.endswith('.pt') and os.path.isfile(file_path):
			if file_name.startswith('D16'):
				continue
			print(f'Loading {file_name}...')
			model_name = os.path.splitext(file_name)[0]

			is_equivarient = False
			if any(char.isdigit() for char in model_name):
				is_equivarient = True

			feature_model = load_model(file_path, model_name, is_equivarient)

			perturbed_data = os.path.join(args.perturbed_data_dir, f'onepixel_attack_results_{model_name}.h5')
			original_images, perturbed_images = load_perturbed_data(args.data_path, perturbed_data)

			dataloader = DataLoader(original_images, batch_size=len(original_images), shuffle=False)
			orig_images = next(iter(dataloader))[0]

			original_latent_space_representation = get_latent_space_represenatation(feature_model, orig_images)
			perturbed_latent_space_representation = get_latent_space_represenatation(feature_model, perturbed_images)


			original_latent_space_representation = original_latent_space_representation.cpu().detach().numpy()
			perturbed_latent_space_representation = perturbed_latent_space_representation.cpu().detach().numpy()


			#Calculate the distance between the original and perturbed images in the latent space
			distance = np.linalg.norm(original_latent_space_representation - perturbed_latent_space_representation, axis=1)

			#mean distance
			print(f"Mean distance between original and perturbed images in the latent space for {model_name}: ", np.mean(distance))
			
			mean_perturbed[model_name] = np.mean(distance)

	mean_perturbed_key = list(mean_perturbed.keys())
	mean_perturbed_value = list(mean_perturbed.values())
	combined_array = np.vstack((key_array, value_array)).T
	print(combined_array)
	np.save('/work/GDL/mean_perturbed.npy', combined_array)

	mean_nosiy_25 = {}
	mean_nosiy_50 = {}

	for file_name in os.listdir(args.model_dir):
		file_path = os.path.join(args.model_dir, file_name)
		if file_name.endswith('.pt') and os.path.isfile(file_path):
			print(f'Loading {file_name}...')
			model_name = os.path.splitext(file_name)[0]

			is_equivarient = False
			if any(char.isdigit() for char in model_name):
				is_equivarient = True

			feature_model = load_model(file_path, model_name, is_equivarient)	

			original_images, noisy_images_25, noisy_images_50 = load_noisy_data(args.data_path, args.noisy_data_25, args.noisy_data_50)

			original_images = DataLoader(original_images, batch_size=128, shuffle=False)
			noisy_images_25 = DataLoader(noisy_images, batch_size=128, shuffle=False)
			noisy_images_50 = DataLoader(noisy_images, batch_size=128, shuffle=False)

			original_latent_space_representation = []
			noisy_25_latent_space_representation = []
			noisy_50_latent_space_representation = []
			for img, label, _, _ in original_images:
				original_latent_space_representation.append(get_latent_space_represenatation(feature_model, img))

			original_latent_space_representation = np.concatenate(original_latent_space_representation, axis=0)
			
			for img, label, _, _ in noisy_images_25:
				noisy_25_latent_space_representation.append(get_latent_space_represenatation(feature_model, img))
			
			noisy_25_latent_space_representation = np.concatenate(noisy_25_latent_space_representation, axis=0)

			for img, label, _, _ in noisy_images_50:
				noisy_50_latent_space_representation.append(get_latent_space_represenatation(feature_model, img))
			
			noisy_50_latent_space_representation = np.concatenate(noisy_50_latent_space_representation, axis=0)

			mean_nosiy_25[model_name] = np.mean(np.linalg.norm(original_latent_space_representation - noisy_25_latent_space_representation, axis=1))
			mean_nosiy_50[model_name] = np.mean(np.linalg.norm(original_latent_space_representation - noisy_50_latent_space_representation, axis=1))

			print(f"Mean distance between original and 25% noisy images in the latent space for {model_name}: ", mean_nosiy_25[model_name])
			print(f"Mean distance between original and 50% noisy images in the latent space for {model_name}: ", mean_nosiy_50[model_name])
			

	##save dict as np array
	

	mean_nosiy_25_key = list(mean_nosiy_25.keys())
	mean_nosiy_25_value = list(mean_nosiy_25.values())
	combined_array = np.vstack((key_array, value_array)).T
	print(combined_array)
	np.save('/work/GDL/mean_nosiy_25.npy', combined_array)

	mean_nosiy_50_key = list(mean_nosiy_50.keys())
	mean_nosiy_50_value = list(mean_nosiy_50.values())
	combined_array = np.vstack((key_array, value_array)).T
	print(combined_array)
	np.save('/work/GDL/mean_nosiy_50.npy', combined_array)

	#sort mean dict
	#mean = dict(sorted(mean.items(), key=lambda item: item[1]))

	mean_cnn = mean['CNN']

	mean_cn = {k : v for k, v in mean.items() if 'C' in k}
	mean_cn = dict(sorted(mean_cn.items()))
	mean_dn = {k : v for k, v in mean.items() if 'D' in k}
	mean_dn = dict(sorted(mean_dn.items()))

	##save dict as csv



	## plot and save graph

	sns.lineplot(data=mean, palette='hsv', marker='o', markersize=5)
	plt.title(f'Mean Distance between Original and Perturbed Images in the Latent Space')
	plt.xlabel('Model')
	plt.ylabel('Mean Distance')
	plt.savefig(os.path.join('/work/GDL/', f'mean_distance.png'), bbox_inches='tight',dpi=300)
