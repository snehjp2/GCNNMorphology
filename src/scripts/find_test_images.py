from test import load_models, compute_metrics
from dataset import Galaxy10DECalsTest
from models import model_dict
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from typing import Tuple, List, Optional, Dict


class CorrectClassified:
    @staticmethod
    def correct_classified_indices(test_dataset: Galaxy10DECalsTest, model: nn.Module,
                                   device: str = 'cuda') -> np.ndarray:
        """
        Get indices of correctly classified samples in the test dataset.

        Args:
            test_dataset (Galaxy10DECalsTest): The test dataset.
            model (nn.Module): The trained model.
            device (str): The device to run the model on.

        Returns:
            np.ndarray: Indices of correctly classified samples.
        """

        test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        y_pred, y_true = [], []
        model = nn.DataParallel(model)
        model.to(device)
        model.eval()

        for batch in tqdm(test_dataloader, unit="batch", total=len(test_dataloader), desc="Evaluating model"):
            inputs, labels, _, _ = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            pred_labels = torch.argmax(outputs, dim=-1).cpu().numpy()

            y_pred.extend(pred_labels)
            y_true.extend(labels.cpu().numpy())

        y_pred, y_true = np.asarray(y_pred), np.asarray(y_true)
        indices = np.where(y_pred == y_true)[0]

        return indices


def main(model_dir: str, test_dataset: Galaxy10DECalsTest) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate models on the test dataset and create a subset of correctly classified samples.

    Args:
        model_dir (str): Directory where the models are stored.
        test_dataset (Galaxy10DECalsTest): The test dataset.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Subset of images, labels, and indices.
    """

    trained_models = load_models(model_dir)
    print('All Models Loaded!')

    indices_dict: Dict[str, Optional[np.ndarray]] = {model_name: None for model_name in trained_models.keys()}

    for model_name, model in tqdm(trained_models.items(), desc="Processing models"):
        indices_dict[model_name] = CorrectClassified.correct_classified_indices(test_dataset, model)

    # Find the intersection of all the indices
    intersection = list(set.intersection(*map(set, [idx for idx in indices_dict.values() if idx is not None])))
    intersection_set = set(intersection)  # Create a set for faster lookups

    subset_images, subset_labels, subset_idxs = [], [], []

    for idx in tqdm(range(len(test_dataset)), desc="Creating subset"):
        if idx in intersection_set:
            image, label, _, _ = test_dataset[idx]
            subset_images.append(image.cpu().numpy())
            subset_labels.append(label.cpu().numpy())
            subset_idxs.append(idx)

    print(f"Number of images in the subset: {len(subset_images)}")
    print(f"Number of images in original test set: {len(test_dataset)}")

    return np.array(subset_images), np.array(subset_labels), np.array(subset_idxs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Path to Models and Data')
    parser.add_argument('--model_dir', required=True, help='Location of the model directory with trained models')
    parser.add_argument('--data_path', required=True, help='Location of the test data .h5 file')
    parser.add_argument('--output_name', required=True, help='Name of the output file')
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        transforms.Resize(255)
    ])

    test_dataset = Galaxy10DECalsTest(str(args.data_path), transform)
    print("Test Dataset Loaded!")

    subset_images, subset_labels, subset_idxs = main(args.model_dir, test_dataset)

    np.save(f'{args.model_dir}/{args.output_name}_idxs.npy', subset_idxs)
