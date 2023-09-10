import torch
import torch.nn.functional as F
import cv2
import numpy as np
import torch
from models import model_dict
from dataset import Galaxy10DECalsTest
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
import warnings
import argparse

device = ('cuda' if torch.cuda.is_available() else 'cpu')

class GradCAM:
    def __init__(self, model, equiv=False):
        self.model = model
        self.model.to(device)
        self.model.eval()
        self.feature = None
        self.gradient = None
        self.equiv = equiv

    def save_gradient(self, module, grad_input, grad_output):
        print("backward hook")
        self.gradient = grad_output[0]

    def __call__(self, x, index=None):
        if self.equiv:
            final_layer = list(dict(self.model.named_children()).values())[-4].conv
        else:
            final_layer = list(dict(self.model.named_children()).values())[-3].conv         
            
        final_layer.register_forward_hook(self.save_feature)
        final_layer.register_backward_hook(self.save_gradient)


        # Forward
        logits = self.model(x)
        if index == None:
            index = np.argmax(logits.cpu().data.numpy())

        # Create a one-hot tensor for the predicted class
        one_hot = np.zeros((1, logits.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(device) * logits)
        
        # Backward
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        # Get pooled gradients
        weights = torch.mean(self.gradient, dim=(2, 3), keepdim=True)
        activation_maps = F.relu(torch.sum(weights * self.feature, dim=1)).squeeze()

        return activation_maps

    def save_feature(self, module, input, output):
        print('forward hook')
        self.feature = output

def visualize(img, cam):
    # Normalize the CAM so values are between 0 and 1
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    cam = np.uint8(cam * 255)
    resized_cam = cv2.resize(cam, (255,255))
    heatmap = cv2.applyColorMap(resized_cam, cv2.COLORMAP_JET)
    
    # Convert your torch tensor image to numpy
    img = img.cpu().numpy()
    img = np.uint8(img * 255)  # Convert from [0,1] to [0,255]
    
    out = cv2.addWeighted(img, 0.8, heatmap, 0.2, 0)
    return out


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(255)
])
og_data = Galaxy10DECalsTest('/n/holystore01/LABS/iaifi_lab/Users/spandya/data/GCNN_imbalanced/test_data_imbalanced.hdf5', transform=transform) ## removed transform
snr_data = Galaxy10DECalsTest('/n/holystore01/LABS/iaifi_lab/Users/spandya/data/GCNN_imbalanced/test_data_imbalanced_res_25.hdf5', transform=transform)

index = np.random.randint(0, len(og_data))
og_image, og_label, _, _ = og_data[index]
snr_image, snr_label, _, _ = snr_data[index]

og_image = og_image.unsqueeze(0)
snr_image = snr_image.unsqueeze(0)

def main(args):

    model_str = args.model
    model = model_dict[model_str]()
    model.load_state_dict(torch.load(f'/n/holystore01/LABS/iaifi_lab/Users/spandya/new_icml/new_models/{model_str}.pt', map_location=device))

    grad_cam = GradCAM(model=model, equiv=True)
    activation_map_og = grad_cam(og_image.to(device))
    activation_map_snr = grad_cam(snr_image.to(device))

    og_img = og_image.squeeze().permute(1,2,0)
    snr_img = snr_image.squeeze().permute(1,2,0) 
    og_result = visualize(og_img, activation_map_og.cpu().data.numpy())
    snr_result = visualize(snr_img, activation_map_snr.cpu().data.numpy())
    og_output = model(og_image.to(device))
    snr_output = model(snr_image.to(device))
    _ , og_predicted_class = torch.max(og_output, 1)
    _ , snr_predicted_class = torch.max(snr_output, 1)


    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plt.suptitle(f'Grad-CAM Original vs. Noisy Image. Index: {index}')
    cax1 = ax[0].imshow(og_result, cmap='jet')
    cax2 = ax[1].imshow(snr_result, cmap='jet')
    plt.colorbar(cax1, ax=ax[0])  
    plt.colorbar(cax2, ax=ax[1]) 
    ax[0].set_title(f'True Label: {og_label.item()}, Predicted Label: {og_predicted_class.item()}')
    ax[1].set_title(f'True Label: {snr_label.item()}, Predicted Label: {snr_predicted_class.item()}')
    ax[0].axis('off')
    ax[1].axis('off')

    plt.tight_layout()
    plt.savefig('gradcam.png')
    plt.show()
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='GradCAM')
    parser.add_argument('--model', type=str, default='C1', help='Model to use')
    args = parser.parse_args()
    main(args)