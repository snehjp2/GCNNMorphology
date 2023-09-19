import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse

from torch.utils.data import DataLoader
from sklearn.manifold import Isomap, TSNE
import umap
from models import model_dict
from escnn import nn as escnn_nn
from escnn import gspaces
from dataset import Galaxy10DECalsTest
from torchvision import transforms
import seaborn as sns



class FeatureModel(nn.Module):
	def __init__(self, base_model: nn.Module, is_equivarient):
		super(FeatureModel, self).__init__()
		self.features = torch.nn.Sequential(*list(base_model.children())[:-1])
		self.linear_layers = torch.nn.Sequential(*list(base_model.children())[-1:])
		print(self.linear_layers)
		self.is_equivarient = is_equivarient
		if is_equivarient:
			self.in_type = base_model.input_type

	def forward(self, x):
		if self.is_equivarient:
			x = escnn_nn.GeometricTensor(x, self.in_type)
		features = self.features(x)
		if self.is_equivarient:
			features = features.tensor
		features = features.reshape(features.shape[0], -1)
		output = self.linear_layers(features)
		return features, output


def load_model(model_path: str, model_name: str, is_equivarient: bool):
	"""
	Load model from .pt file.
	"""
	model = model_dict[str(model_name)]()
	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	ckpt = torch.load(model_path, map_location=device)

	model.load_state_dict(ckpt)

	feature_model = FeatureModel(model, is_equivarient)
	feature_model.eval()
	return feature_model

def generate_embedding_vector(model: nn.Module, test_dataloader: DataLoader):
	"""
	Generate embedding vector for each image in test set.
	"""
	embedding_vector = []
	labels_vector = []
	i = 0
	model.to(device)
	for images, labels,  angle, redshift  in test_dataloader:
		print(f'Batch {i}')
		images = images.to(device)
		labels_vector.append(labels.numpy())
		embedding_vector.append(model(images).detach().cpu().numpy())
		i += 1
	embedding_vector = np.concatenate(embedding_vector, axis=0)
	labels_vector = np.concatenate(labels_vector, axis=0)
	print(f'Embedding Vector Shape: {embedding_vector.shape}')
	print(f'Labels Vector Shape: {labels_vector.shape}')

	return embedding_vector, labels_vector

def plot_isomap(data: np.ndarray, labels: np.ndarray, save_dir: str, model_name: str):
	"""
	Plot isomap of embedding vector.
	"""

	classes = ('Disturbed Galaxies', 'Merging Galaxies', 
        'Round Smooth Galaxies', 'In-between Round Smooth Galaxies', 
        'Cigar Shaped Smooth Galaxies', 'Barred Spiral Galaxies', 
        'Unbarred Tight Spiral Galaxies', 'Unbarred Loose Spiral Galaxies', 
        'Edge-on Galaxies without Bulge', 'Edge-on Galaxies with Bulge')


	embedding = Isomap(n_components=2)
	embedding.fit(data)
	embedding_vector = embedding.transform(data)

	unique_labels = np.unique(labels)
	colors = [plt.cm.jet(i/float(len(unique_labels)-1)) for i in range(len(unique_labels))]
 
	### save data to npy file
 
	np.save(os.path.join(save_dir, f'isomap-{model_name}.npy'), embedding_vector)
	np.save(os.path.join(save_dir, f'labels-{model_name}.npy'), labels)
	
	print('Data Saved')
 
	# for i, label in enumerate(unique_labels):
    # # Extract the points that have this label
	# 	label_points = embedding_vector[labels == label] 
	# 	plt.scatter(label_points[:, 0], label_points[:, 1], color=colors[i], label=classes[i], s=1)
	
	# #plt.figure(figsize=(10,6))
	# plt.legend(loc='best',fontsize='xx-small')
	# plt.savefig(os.path.join(save_dir, f'isomap-{model_name}.png'))


def plot_tsne(data: np.ndarray, labels: np.ndarray, save_dir: str, model_name: str):
	"""
	Plot tsne of embedding vector.
	"""

	classes = ('Disturbed Galaxies', 'Merging Galaxies', 
        'Round Smooth Galaxies', 'In-between Round Smooth Galaxies', 
        'Cigar Shaped Smooth Galaxies', 'Barred Spiral Galaxies', 
        'Unbarred Tight Spiral Galaxies', 'Unbarred Loose Spiral Galaxies', 
        'Edge-on Galaxies without Bulge', 'Edge-on Galaxies with Bulge')

	tsne = TSNE(n_components=2, random_state=42)
	tsne_vector = tsne.fit_transform(data)

	df = pd.DataFrame({
		'x': tsne_vector[:,0],
		'y': tsne_vector[:,1],
		'class': [classes[i] for i in labels]
	})

	plt.figure(figsize=(10,8))
	sns.scatterplot(data=df, x='x', y='y', hue='class', palette='hsv')
	plt.title(f'TSNE Visualization of Galaxy10DECals Dataset using {model_name}')
	plt.savefig(os.path.join(save_dir, f'tsne-{model_name}.png'))


def plot_umap(data: np.ndarray, labels: np.ndarray, save_dir:str, model_name:str):
	"""
	Plot umap of embedding vector.
	"""

	classes = ('Disturbed Galaxies', 'Merging Galaxies', 
        'Round Smooth Galaxies', 'In-between Round Smooth Galaxies', 
        'Cigar Shaped Smooth Galaxies', 'Barred Spiral Galaxies', 
        'Unbarred Tight Spiral Galaxies', 'Unbarred Loose Spiral Galaxies', 
        'Edge-on Galaxies without Bulge', 'Edge-on Galaxies with Bulge')

	reducer = umap.UMAP(random_state=42)
	umap_vector = reducer.fit_transform(data)

	df = pd.DataFrame({
		'x': umap_vector[:,0],
		'y': umap_vector[:,1],
		'class': [classes[i] for i in labels]
	})
 
	### save dataframe to npy file

	plt.figure(figsize=(10,8))
	sns.scatterplot(data=df, x='x', y='y', hue='class', palette='hsv') 
	plt.title(f'UMAP Visualization of Galaxy10DECals Dataset using {model_name}')
	plt.savefig(os.path.join(save_dir, f'umap-{model_name}.png'))



	



if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Plot Isomap')

	parser.add_argument('--model_path', type=str, required=True, help='Path to model')
	parser.add_argument('--model_name', type=str, required=True, help='Name of model')
	parser.add_argument('--data_path', type=str, required=True, help='Path to data')
	parser.add_argument('--save_dir', type=str, required=True, help='Path to save directory')

	args = parser.parse_args()

	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	#checks if model_name has number in it
	is_equivarient = False
	if any(char.isdigit() for char in args.model_name):
		is_equivarient = True
	feature_model = load_model(args.model_path, args.model_name, is_equivarient)
	transform = transforms.Compose([
        transforms.ToTensor(),
		transforms.Resize(255),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
	test_dataset = Galaxy10DECalsTest(args.data_path, transform)
	print("Test Dataset Loaded!")
	print(f'Test Dataset Length: {len(test_dataset)}')
	test_dataloader = DataLoader(test_dataset, batch_size = 64, shuffle=True)
	embedding_vector, labels_vector = generate_embedding_vector(feature_model, test_dataloader)
	plot_isomap(embedding_vector, labels_vector, args.save_dir, args.model_name)
	# plot_tsne(embedding_vector, labels_vector, args.save_dir, args.model_name)
	# plot_umap(embedding_vector, labels_vector, args.save_dir, args.model_name)
