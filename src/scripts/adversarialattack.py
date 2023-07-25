from torch.utils.data import DataLoader
import argparse
import torch
import os
from dataset import Galaxy10DECalsTest
from torchvision import transforms
import matplotlib.pyplot as plt
from train import set_all_seeds

class OnePixelAttack:
    def __init__(self, pixel_value=1.0):  # pixel_value=1.0 corresponds to white in [0, 1] range
        """
        pixel_value: a float indicating the value to change the pixel to. 
        By default, it changes the pixel to white.
        """
        self.pixel_value = pixel_value

    def __call__(self, image):
        """
        image: a PyTorch tensor in (C, H, W) format
        """
        # Randomly select the pixel to be attacked.
        c, h, w = image.shape
        #rand_c = torch.randint(0, c, (1,)).item()
        rand_h = torch.randint(0, h, (1,)).item()
        rand_w = torch.randint(0, w, (1,)).item()

        # Change the pixel value to white.
        image[:, rand_h, rand_w] = self.pixel_value

        return image




if __name__ == '__main__':

    set_all_seeds(42)

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    
    parser = argparse.ArgumentParser(description = 'Path to Models and Data')
    parser.add_argument('--model_path', metavar = 'model_path', required=True,
                    help='Location of the model directory')
    
    parser.add_argument('--data_path', metavar = 'data_path', required=True, help='Location of the test data file')
    
    parser.add_argument('--output_name', metavar = 'output_name', required=True, help='Name of the output file')

    args = parser.parse_args()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(255),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        OnePixelAttack()
    ])
    
    test_dataset = Galaxy10DECalsTest(str(args.data_path), transform)
    print("Test Dataset Loaded!")
    test_dataloader = DataLoader(test_dataset, batch_size = 128, shuffle=True)
    
    #main(str(args.model_path), str(args.output_name))
	
    image, label, angle, redshift = next(iter(test_dataloader))

    #plot the image using matlplotlib
    plt.imshow(image[0].permute(1, 2, 0))
    plt.show()

    image, label, angle, redshift = next(iter(test_dataloader))

    #plot the image using matlplotlib
    plt.imshow(image[0].permute(1, 2, 0))
    plt.show()
    
    image, label, angle, redshift = next(iter(test_dataloader))

    #plot the image using matlplotlib
    plt.imshow(image[0].permute(1, 2, 0))
    plt.show()

        
