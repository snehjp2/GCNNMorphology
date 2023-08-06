import numpy as np 
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from dataset import Galaxy10DECalsTest
from test import load_models
from torchvision import transforms
import matplotlib.pyplot as plt
from models import model_dict, GeneralSteerableCNN
from train import set_all_seeds
from find_test_images import correct_classified_indices
import random
import argparse
import time
import os
from tqdm import tqdm
import h5py 

def show(img):
    npimg = img.cpu().numpy()
    plt. imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

def load_model(model_name, model_dir_path, device = 'cuda'):
    file_path = os.path.join(model_dir_path,f"{model_name}.pt")
    model = model_dict[str(model_name)]()
    model.eval()
    model.load_state_dict(torch.load(file_path, map_location=device))

    return model

def tell(base_network, img, label, target_label=None):
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
 
def visualize_perturbation(base_network, p, img, label,target_label=None):
    p_img = perturb(p, img)
    print("Perturbation:", p)
    #show(p_img)
    tell(base_network, p_img, label, target_label)

def perturb(p, img):
    img = img.to(device)
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

def evaluate(candidates, img, label, model):
    
    model = model.to(device)
    img = img.to(device)
    label = label.to(device)
    
    preds = []
    model.train(False)
    perturbed_img = []
 
    perturbed_img = [perturb(xs, img) for xs in candidates]
    perturbed_img = torch.stack(perturbed_img).to(device)  
    
    with torch.no_grad():
        for i in range(0, len(perturbed_img), 128):
            data = perturbed_img[i:i+128] if i+128 < len(perturbed_img) else perturbed_img[i:]
            data = data.to(device)
            output = F.softmax(model(data), dim=1)
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

def attack(model, img, true_label, iters=100, pop_size=400):
    # Targeted: maximize target_label if given (early stop > 50%)
    # Untargeted: minimize true_label otherwise (early stop < 5%)
    model = model.to(device)
    img = img.to(device)
    true_label = true_label.to(device)
    
    candidates = np.random.random((pop_size,5))
    candidates[:,2:5] = np.clip(np.random.normal(0.5, 0.5, (pop_size, 3)), 0, 1)
    label = true_label
    
    fitness = evaluate(candidates, img, label, model)
    fitness_history = []
    
    def is_success():
        return fitness.min() < 0.05

    is_missclassified = False

    for iteration in range(iters):
        # Early Stopping
        if is_success():
            break

        # Generate new candidate solutions
        new_gen_candidates = evolve(candidates, strategy="resample")
        
        # Evaluate new solutions
        new_gen_fitness = evaluate(new_gen_candidates, img, label, model)
       
        # Replace old solutions with new ones where they are better
        successors = new_gen_fitness < fitness
        candidates[successors] = new_gen_candidates[successors]
        fitness[successors] = new_gen_fitness[successors]
        best_idx =  fitness.argmin()
        fitness_history.append(fitness[best_idx])

        best_candidate = candidates[best_idx]
        p_img = perturb(best_candidate, img)
        out = F.softmax(model(p_img.unsqueeze(0)).squeeze(), dim=0)
        pred_label = out.detach().cpu().numpy().argmax()
        if pred_label != true_label and not is_missclassified:
            print(f"First missclassfication at iteration {iteration}")
            is_missclassified = True
            break

    
    best_idx = fitness.argmin()
    best_solution = candidates[best_idx]
    best_score = fitness[best_idx]

    perturbed_img = perturb(best_solution, img)

    return (is_success() or is_missclassified), best_solution, best_score, perturbed_img, pred_label, iteration+1, fitness_history #it starts at 0


def main(model, test_dataset, args):

    perturbed_images = []
    labels = []
    pred_labels = []
    indices = []
    iteration_counter = []
    
    for i in tqdm(range(len(test_dataset))):
        img, label, _, _ = test_dataset[i]
        print(f"Image {i} with label {label}")
        img = img.to(device)
        
        is_success, best_solution, best_score, perturbed_img, pred_label, iterations, fitness_history = attack(model, img, label, iters=200, pop_size=400)
        if is_success:
            print(f"Success at iteration {iterations}")
            perturbed_img = perturbed_img.cpu().numpy()
            perturbed_images.append(perturbed_img)
            labels.append(label)
            iteration_counter.append(iterations)
            pred_labels.append(pred_label)
            indices.append(i)

    f = h5py.File(os.path.join(args.output_dir,f"onepixel_attack_results_{args.model_name}.h5"),'w')

    images = np.concatenate(perturbed_images)
    labels_out = np.asarray(labels)
    indices = np.asarray(indices)
    iteration_counter = np.asarray(iteration_counter)
    image_dataset = f.create_dataset(f"images", np.shape(images), data=images, compression='gzip', chunks=True)
    label_dataset = f.create_dataset(f"labels", np.shape(labels_out), data=labels_out, compression='gzip', chunks=True)
    pred_label_dataset = f.create_dataset(f"pred_labels", np.shape(pred_labels), data=pred_labels, compression='gzip', chunks=True)
    indices_dataset = f.create_dataset(f"indices", np.shape(indices), data=indices, compression='gzip', chunks=True)
    iteration_counter_dataset = f.create_dataset(f"iterations", np.shape(iteration_counter), data=iteration_counter, compression='gzip', chunks=True)
    f.close()
 

if __name__ == '__main__':

    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description = 'One pixel attack on a model')

    parser.add_argument('--model_name', metavar= 'model_name', required=True, help='Name of the model')
    parser.add_argument('--model_dir_path', metavar = 'config', required=True,help='path to model directory')
    parser.add_argument('--data_path', metavar = 'data_path', required=True, help='Location of the test data file')
    parser.add_argument('--output_dir', metavar = 'output_dir', required=True, help='output directory')
    parser.add_argument('--idx_path', metavar='idx_path', required=True, help='directory of the indices')
    args = parser.parse_args()
    
    model = load_model(args.model_name, args.model_dir_path)
    print('Model Loaded')

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            transforms.Resize(255)
        ])
    
    custom_idxs = np.load(str(args.idx_path), allow_pickle=True)
    test_dataset = Galaxy10DECalsTest(str(args.data_path), transform, custom_idxs=custom_idxs)
    print(len(test_dataset))
    print('Dataset Loaded')

    main(model, test_dataset, args)