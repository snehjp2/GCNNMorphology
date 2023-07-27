import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models import model_dict
from train import set_all_seeds
import random


def show(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

def tell(base_network, img, label, model, target_label=None):
    output = F.softmax(base_network(img.unsqueeze(0))[1].squeeze(), dim=0)
    LABELS = ('spiral', 'elliptical', 'merger')
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
    show(p_img)
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
    with torch.no_grad():
        for i, xs in enumerate(candidates):
            p_img = perturb(xs, img)
            preds.append(F.softmax(base_network(p_img.unsqueeze(0))[1].squeeze(), dim=0)[int(label)].item())
 
    return np.array(preds)

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

def attack(model, img, true_label, target_label=None, iters=100, pop_size=400, verbose=True):
    # Targeted: maximize target_label if given (early stop > 50%)
    # Untargeted: minimize true_label otherwise (early stop < 5%)

    candidates = np.random.random((pop_size,5))
    candidates[:,2:5] = np.clip(np.random.normal(0.5, 0.5, (pop_size, 3)), 0, 1)
    is_targeted = target_label is not None
    label = target_label if is_targeted else true_label
    fitness = evaluate(model, candidates, img, label, model)
    
    def is_success():
        return (is_targeted and fitness.max() > 0.5) or ((not is_targeted) and fitness.min() < 0.05)
    
    for iteration in range(iters):
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
    best_solution = candidates[best_idx]
    best_score = fitness[best_idx]
    
    if verbose:
        visualize_perturbation(model, best_solution, img, true_label, model, target_label)

    perturbed_img = perturb(best_solution, img)
    return is_success(), best_solution, best_score, perturbed_img, iteration+1 #it starts at 0

def load_model(model_name, path):
	model = model_dict[model_name]
	model.eval()
	model.load_state_dict(torch.load(path, map_location=device))
	
	return model

if __name__ == '__main__':

	device = ('cuda' if torch.cuda.is_available() else 'cpu')

	set_all_seeds(42)

	parser = argparse.ArgumentParser(description = 'One pixel attack on a model')

	parser.add_argument('--model', metavar = 'model', required=True, help='Name of the model to use')
	parser.add_argument('--model_path', metavar = 'config', required=True,
                    help='path to model')

	parser.add_argument('--data_path', metavar = 'data_path', required=True, help='Location of the test data file')


	args = parser.parse_args()

	model = load_model(args.model, args.model_path)

	transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            transforms.Resize(255)
        ])

	test_dataset = Galaxy10DECalsTest(str(args.data_path), transform)
	i = random.randint(0, len(test_dataset))
	img, label, angle, redshift = test_dataset[i]

	img = img.to(device)
	model = model.to(device)

	is_success, best_solution, best_score, perturbed_img, iterations = attack(model, img, label, target_label=None, iters=100, pop_size=400, verbose=True)

	print("Attack Success:", is_success)
	print("Best Solution:", best_solution)
	print("Best Score:", best_score)
	print("Iterations:", iterations)

	plt.title(f'{model_name} Perturbed image')
	plt.savefig(os.path.join('/work/GDL/purvik/', f"perturbed_image_{model_name}.png"), bbox_inches='tight')
	plt.close()