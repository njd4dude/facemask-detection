import os 
from torch.utils.data import Dataset
from PIL import Image


class CustomImageDataset(Dataset):
    def __init__(self, image_dir, labels, transform=None):
        self.image_dir = image_dir
        self.labels = labels  # Labels should be a list
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))  # Sort to maintain order

        # Ensure label count matches image count
        if len(self.image_files) != len(self.labels):
            raise ValueError(f"Mismatch: {len(self.image_files)} images but {len(self.labels)} labels.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])

        # Handle missing files
        if not os.path.exists(img_path):
            print(f"Warning: Missing file {img_path}")
            return self.__getitem__((idx + 1) % len(self))  # Skip to next sample

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))  # Skip to next sample

        label = self.labels[idx]

        # Apply transformations safely
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Transform error on {img_path}: {e}")
                return self.__getitem__((idx + 1) % len(self))  # Skip to next sample

        return image, label
