from utils import prepare_datasets, create_dataloaders 
from checks import check_dataset, check_dataloader
import torch
from torchvision import transforms
import os
from collections import defaultdict
from saveSplit import save_images_to_folders

# so that means the factor we have to look out for is the domain of the data between the train and validation since we already know the train/validation split is good.
# another thing to look out for is the cropping of the face is done correctly. Task 3/22: the dataset contains a lot of augmented images so we have to be careful about that.
# one thing i want to try next is creating my own dataset from scratch and see if that makes a difference.

# task 3/25: work on creating the trashCameraQuality dataset and see it how it performs on my camera
# I'm going to take some of images taken from my camera and see how well the model performs on it.

# task: combined both trashCameraQuality and dataset2 or dataset1 

def check_data_leakage(train_dataset, validation_dataset):
    print("\nChecking data leakage...")

    # If dataset uses indices, check for overlap
    if hasattr(train_dataset, 'indices') and hasattr(validation_dataset, 'indices'):
        train_indices = set(train_dataset.indices)
        validation_indices = set(validation_dataset.indices)

        if train_indices & validation_indices:
            print("⚠️ Data leakage detected: Overlapping indices found!")
        else:
            print("✅ No data leakage detected based on indices.")


   

def save_subset_images(subset, output_folder):
    # Make sure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Define a transform to convert tensor to PIL image (assuming input tensor values are in [0, 1])
    to_pil_image = transforms.ToPILImage()

    # Dictionary to count label distribution
    label_distribution = defaultdict(int)

    # Iterate over the first 100 images in the subset
    for idx in range(min(50, len(subset))):  # Limit to 100 images or the total length of the subset
        image, label = subset[idx]
        
        # Update the label distribution
        label_distribution[label] += 1
        
        # If the image is normalized, reverse the normalization (example for ImageNet normalization)
        # Adjust these values based on how you normalized your images
        if isinstance(image, torch.Tensor):  # Make sure it's a tensor
            image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        
        # Convert tensor to PIL image
        pil_image = to_pil_image(image)
        
        # Define a filename for the image
        filename = f"{output_folder}/image_{idx}_label_{label}.png"
        
        # Save the image to the output folder
        pil_image.save(filename)
        print(f"Saved {filename}")

    # Print the label distribution
        
    print("\nLabel Distribution (first 100 images):")
    for label, count in label_distribution.items():
        print(f"Label {label}: {count} images")


if __name__ == "__main__":
    print("Prepare Dataset initiated...")

    dataset_folder_name = "dataset2-trashCameraQuality"
    train_dataset, validation_dataset = prepare_datasets(dataset_base_path=os.path.join("Datasets", dataset_folder_name))

    print("\n\n DATASETS CHECK----------")
    print("\ntrain_dataset")
    check_dataset(train_dataset)
    print("\nvalidation_dataset")
    check_dataset(validation_dataset)
    
    # check for data leakage
    check_data_leakage(train_dataset, validation_dataset)

    # Save images from the train and validation datasets for sanity check
    save_subset_images(train_dataset, f"DatasetSplitImages/{dataset_folder_name}/train")
    save_subset_images(validation_dataset, f"DatasetSplitImages/{dataset_folder_name}/validation")

    train_dataloader, validation_dataloader = create_dataloaders(train_dataset, validation_dataset)

    print("\n\n DATALOADERS CHECK----------------")
    print("\ntrain_dataloader")
    check_dataloader(train_dataloader)
    print("\nvalidation_dataloader")
    check_dataloader(validation_dataloader)

        
    # Create Dataloaders directory if it doesn't exist
    output_dataloader_dir = f"Dataloaders/{dataset_folder_name}"
    os.makedirs(output_dataloader_dir, exist_ok=True)  # Creates the directory if it doesn't exist

    # Save the dataloaders
    torch.save(train_dataloader, os.path.join(output_dataloader_dir, "train_dataloader.pth"))
    torch.save(validation_dataloader, os.path.join(output_dataloader_dir, "validation_dataloader.pth"))

    print(f"\nSaved train and validation dataloaders to {output_dataloader_dir}")

# Dataset 3 is not good for generalizing probably because of too many duplciate images its learning the colors rather than the actual pattern of mask. 