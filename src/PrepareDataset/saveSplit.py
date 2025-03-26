import os
import shutil
from torchvision.utils import save_image
from utils import prepare_datasets

def save_images_to_folders(dataset, folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

    for i in range(len(dataset)):
        image, label = dataset[i]
        label_folder = os.path.join(folder_path, str(label))
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)
        image_path = os.path.join(label_folder, f"image_{i}.png")
        save_image(image, image_path)

if __name__ == "__main__":
    # Example usage:
    dataset_base_path = 'Datasets/test'
    train_dataset, validation_dataset = prepare_datasets(dataset_base_path)

    train_folder = 'train_images'
    validation_folder = 'validation_images'

    print("Saving Training Dataset images...")
    save_images_to_folders(train_dataset, train_folder)

    print("Saving Validation Dataset images...")
    save_images_to_folders(validation_dataset, validation_folder)