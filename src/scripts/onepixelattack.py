import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from dataset import Galaxy10DECalsTest
from test import load_models
from torchvision import transforms
import matplotlib.pyplot as plt
from models import model_dict
from train import set_all_seeds
import random
import argparse
import time
import os


def show(img):
	npimg = img.cpu().numpy()
	plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
  


def tell(base_network, img, label, model, target_label=None):
	output = F.softmax(base_network(img.unsqueeze(0)).squeeze(), dim=0)
	LABELS =  ('Disturbed Galaxies', 'Merging Galaxies', 'Round Smooth Galaxies', 
			   'In-between Round Smooth Galaxies','Cigar Shaped Smooth Galaxies',
			   'Barred Spiral Galaxies', 'Unbarred Tight Spiral Galaxies',
			   'Unbarred Loose Spiral Galaxies', 'Edge-on Galaxies without Bulge',
			   'Edge-on Galaxies with Bulge')
	print("Incoming label: ", label)
	print("True Label:", LABELS[int(label.item())], "-->", int(label.item()))
	print("Prediction:", LABELS[output.tolist().index(max(output.tolist()))],"-->", output.tolist().index(max(output.tolist())))
	print("Label Probabilities:", output.tolist())
	print("True Label Probability:", output[int(np.argmax(label))].item())
	
	if target_label is not None:
		print("Target Label Probability:", output[int(target_label)].item())
		
def visualize_perturbation(base_network, p, img, label, model, target_label=None):
	p_img = perturb(p, img)
	print("Perturbation:", p)
	#show(p_img)
	tell(base_network, p_img, label, model, target_label)

def perturb(p, img):
	img_size = img.size(1) # C x _H_ x W, assume H == W
	p_img = img.clone()
	xy = (np.round(p[0:2].copy()*img_size, 0).astype(int))
	xy = np.clip(xy, 0, img_size-1)
	rgb = p[2:5].copy()
	rgb = np.clip(rgb, 0, 1)
	p_img[:,xy[0],xy[1]] = torch.from_numpy(rgb)
	# print("Pixel position:",xy)
	# print("Attack intensity in three filters:",rgb)
	
	return p_img

def evaluate(base_network, candidates, img, label, model):
	preds = []
	model.train(False)
	perturbed_img = []
	for i, xs in enumerate(candidates):
		p_img = perturb(xs, img)
		perturbed_img.append(p_img)
	perturbed_img = torch.stack(perturbed_img)
	with torch.no_grad():
		for i in range(0, len(perturbed_img), 128):
			data = perturbed_img[i:i+128] if i+128 < len(perturbed_img) else perturbed_img[i:]
			data = data.to(device)
			output = F.softmax(base_network(data), dim=1)
			preds.append(output[:,int(label)].cpu().numpy())

	'''
	with torch.no_grad():
		for i, xs in enumerate(candidates):
			p_img = perturb(xs, img)
			out = base_network(p_img.unsqueeze(0))
			preds.append(F.softmax(out.squeeze(), dim=0)[int(label)].item())
	'''
 
	return np.concatenate(preds)

def evolve(candidates, F=0.5, strategy="clip"):
	gen2 = candidates.copy()
	num_candidates = len(candidates)
	for i in range(num_candidates):
		x1, x2, x3 = candidates[np.random.choice(num_candidates, 3, replace=False)]
		x_next = (x1 + F*(x2 - x3))
		if strategy == "clip":
			gen2[i] = np.clip(x_next, 0, 1)
		elif strategy == "resample":
			x_oob = np.logical_or((x_next < 0), (1 < x_next))
			x_next[x_oob] = np.random.random(5)[x_oob]
			gen2[i] = x_next
	
	return gen2

def attack(model, img, true_label, model_name, target_label=None, iters=100, pop_size=400, verbose=True):
	# Targeted: maximize target_label if given (early stop > 50%)
	# Untargeted: minimize true_label otherwise (early stop < 5%)
	model = model.to(device)
	candidates = np.random.random((pop_size,5))
	candidates[:,2:5] = np.clip(np.random.normal(0.5, 0.5, (pop_size, 3)), 0, 1)
	is_targeted = target_label is not None
	label = target_label if is_targeted else true_label
	fitness = evaluate(model, candidates, img, label, model)
	
	def is_success():
		return (is_targeted and fitness.max() > 0.5) or ((not is_targeted) and fitness.min() < 0.05)

	is_missclassified = False
	for iteration in range(iters):
		print(f'Iteration {iteration}')
		# Early Stopping
		if is_success():
			break
			
		if fitness.max() < .4 and iteration > 80:
			break
		
		if verbose and iteration % 10 == 0: # Print progress
			print("Target Probability [Iteration {}]:".format(iteration), fitness.max() if is_targeted else fitness.min())
		
		# Generate new candidate solutions
		new_gen_candidates = evolve(candidates, strategy="resample")
		
		# Evaluate new solutions
		new_gen_fitness = evaluate(model, new_gen_candidates, img, label, model)
	   
		# Replace old solutions with new ones where they are better
		successors = new_gen_fitness > fitness if is_targeted else new_gen_fitness < fitness
		candidates[successors] = new_gen_candidates[successors]
		fitness[successors] = new_gen_fitness[successors]
		best_idx = fitness.argmax() if is_targeted else fitness.argmin()
		fitness_history.append(fitness[best_idx])

		best_candidate = candidates[best_idx]
		p_img = perturb(best_candidate, img)
		out = F.softmax(model(p_img.unsqueeze(0)).squeeze(), dim=0)
		pred_label = out.cpu().numpy().argmax()
		if pred_label != true_label and not is_missclassified:
			print(f"First missclassfication at iteration {iteration}")
			is_missclassified = True

	
	best_idx = fitness.argmax() if is_targeted else fitness.argmin()
	best_solution = candidates[best_idx]
	best_score = fitness[best_idx]
	
	
	if verbose:
		visualize_perturbation(model, best_solution, img, true_label, model, target_label)

	perturbed_img = perturb(best_solution, img)

	return is_success(), best_solution, best_score, perturbed_img, iteration+1, fitness_history #it starts at 0


if __name__ == '__main__':

	device = ('cuda' if torch.cuda.is_available() else 'cpu')

	set_all_seeds(42)

	parser = argparse.ArgumentParser(description = 'One pixel attack on a model')

	parser.add_argument('--model_dir_path', metavar = 'config', required=True,
					help='path to model directory')

	parser.add_argument('--data_path', metavar = 'data_path', required=True, help='Location of the test data file')


	args = parser.parse_args()

    models = load_models(args.model_dir_path)

	transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
			transforms.Resize(255)
		])

	test_dataset = Galaxy10DECalsTest(str(args.data_path), transform)
	i = random.randint(0, len(test_dataset))
	img, label, angle, redshift = test_dataset[i]

	img = img.to(device)

	fig, ax = plt.subplots()
	for model_name, model in models.items():

		print(f"Attacking {name}...")

		is_success, best_solution, best_score, perturbed_img, iterations, fitness_history = attack(model, img, label, model_name, target_label=None, iters=100, pop_size=400, verbose=True)
		steps = [x for x in range(len(fitness_history))]
		ax.plot(steps, fitness_history, label=model_name)
		print("Attack Success:", is_success)
		print("Best Solution:", best_solution)
		print("Best Score:", best_score)
		print("Iterations:", iterations)

	ax.set_xlabel('Iteration')
	ax.set_ylabel('Target Probability')
	ax.set_title('Target Probability vs Iteration')
	ax.legend()
	fig.savefig(os.path.join('../../../save_dir/', f"target_probability_vs_iterartion.png"), bbox_inches='tight')
	plt.close(fig)


	fig, ax = plt.subplots()
	ax.imshow(np.transpose(perturbed_img.cpu().numpy(), (1,2,0)), interpolation='nearest') 
	ax.set_title(f'{args.model} Perturbed Image')
	fig.savefig(os.path.join('../../../save_dir/', f"perturbed_image_{model_name}.png"), bbox_inches='tight')
	plt.close(fig)

