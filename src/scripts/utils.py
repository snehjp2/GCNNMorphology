import h5py
import numpy as np

class OnePixelAttack:
    
    def __init__(self, path):
        self.path = path
        with h5py.File(path, "r") as f:
            self.imgs = f['images'][()]
            self.labels = f['labels'][()]
            self.pred_labels = f['pred_labels'][()]
            self.indices = f['indices'][()]
            self.iters = f['iterations'][()]

    def get_image(self):
        return np.array(self.imgs)
    
    def get_label(self):
        return np.array(self.labels)
    
    def get_pred_label(self):
        return np.array(self.pred_labels)
    
    def get_img_idx(self):
        return np.array(self.indices)
    
    def get_iter(self):
        return np.array(self.iters)
    
    def __len__(self):
        return len(self.labels)