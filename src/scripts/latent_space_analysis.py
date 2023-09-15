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
import matplotlib.pyplot as plt
import umap

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

	custom_idxs = np.load("/work/GDL/all_correct_idxs.npy", allow_pickle=True)
	original_images = Galaxy10DECalsTest(original_data_filname, transform, custom_idxs=custom_idxs)

	perturbed_images = perturbed_dataset.get_image()
	num = len(perturbed_dataset)
	perturbed_images = perturbed_images.reshape((num,3 , perturbed_images.shape[1], perturbed_images.shape[2]))
	original_images = data.Subset(original_images, perturbed_dataset.get_img_idx())
	perturbed_images = torch.from_numpy(perturbed_images)
	return original_images, perturbed_images

def get_latent_space_represenatation(model, images):
	
	model = model.to(device)
	images = images.to(device)
	latent_space_representation = model(images)
	return latent_space_representation

	
	
if __name__ == '__main__':

	parser = argparse.ArgumentParser(description = 'Latent Space Analysis')
	parser.add_argument('--data_path', type=str, metavar='original_data_filname', required=True,
						help='Location of original dataset')
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

	
	mean = {}
	median = {}
	std = {}

	for file_name in os.listdir(args.model_dir):
		file_path = os.path.join(args.model_dir, file_name)
		if file_name.endswith('.pt') and os.path.isfile(file_path):
			print(f'Loading {file_name}...')
			model_name = os.path.splitext(file_name)[0]

			is_equivarient = False
			if any(char.isdigit() for char in model_name):
				is_equivarient = True

			feature_model = load_model(file_path, model_name, is_equivarient)

			perturbed_data = os.path.join(args.perturbed_data_dir, f'onepixel_attack_results_{model_name}.h5')
			original_images, perturbed_images = load_data(args.data_path, perturbed_data)

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
			#median distance
			print(f"Median distance between original and perturbed images in the latent space for {model_name}: ", np.median(distance))
			#std error
			print(f"Standard error of distance between original and perturbed images in the latent space for {model_name}: ", np.std(distance))

			mean[model_name] = np.mean(distance)
			median[model_name] = np.median(distance)
			std[model_name] = np.std(distance)

	#sort mean dict
	mean = dict(sorted(mean.items(), key=lambda item: item[1]))
	
	## plot and save graph

	sns.lineplot(data=mean, palette='hsv', marker='o', markersize=5)
	plt.title(f'Mean Distance between Original and Perturbed Images in the Latent Space')
	plt.xlabel('Model')
	plt.ylabel('Mean Distance')
	plt.savefig(os.path.join(args.model_dir, f'mean_distance.png'))

	##UMAP Visualization

	# umap = umap.UMAP(random_state=42)

	# data = np.concatenate((original_latent_space_representation, perturbed_latent_space_representation))
	# labels = next(iter(dataloader))[1].cpu().detach().numpy()
	# labels = np.concatenate((labels, labels))

	# umap_vector = umap.fit_transform(data)

	# classes = ('Disturbed Galaxies', 'Merging Galaxies', 
    #     'Round Smooth Galaxies', 'In-between Round Smooth Galaxies', 
    #     'Cigar Shaped Smooth Galaxies', 'Barred Spiral Galaxies', 
    #     'Unbarred Tight Spiral Galaxies', 'Unbarred Loose Spiral Galaxies', 
    #     'Edge-on Galaxies without Bulge', 'Edge-on Galaxies with Bulge')

	# df = pd.DataFrame({
	# 	'x': umap_vector[:,0],
	# 	'y': umap_vector[:,1],
	# 	'class': [classes[i] for i in labels]
	# })

	# half_length = len(df) // 2
	# df_first_half = df.iloc[:half_length]
	# df_second_half = df.iloc[half_length:]


	# plt.figure(figsize=(10,8))
	# sns.scatterplot(data=df_first_half, x='x', y='y', hue='class', palette='hsv', marker='o', s=100)

	# sns.scatterplot(data=df_second_half, x='x', y='y', hue='class', palette='hsv', marker='x', s=100)
	# plt.title(f'UMAP Visualization of Regular and Perturbed Dataset using {args.model_name}')
	# plt.show()