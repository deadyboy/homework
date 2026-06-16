import torch
from torch.utils.data import Dataset
import cv2
from pathlib import Path


class FacadesDataset(Dataset):
    def __init__(self, list_file):
        """
        Args:
            list_file (string): Path to the txt file with image filenames.
        """
        self.list_file = Path(list_file)
        self.base_dir = self.list_file.parent

        with self.list_file.open('r', encoding='utf-8') as file:
            self.image_filenames = [line.strip() for line in file if line.strip()]
        
    def __len__(self):
        # Return the total number of images
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Get the image filename
        img_name = Path(self.image_filenames[idx])
        img_path = img_name if img_name.is_absolute() else self.base_dir / img_name
        img_color_semantic = cv2.imread(str(img_path))
        if img_color_semantic is None:
            raise FileNotFoundError(f'Could not read image: {img_path}')

        image = torch.from_numpy(img_color_semantic).permute(2, 0, 1).float()/255.0 * 2.0 -1.0
        image_rgb = image[:, :, :256]
        image_semantic = image[:, :, 256:]
        return image_rgb, image_semantic
