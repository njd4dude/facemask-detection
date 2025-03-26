import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
matplotlib.use("Agg")

def preview_images(folders, base_dir, num_images=10):
    # Function to sort filenames as numbers
    def numeric_sort(file_name):
        return int(''.join(filter(str.isdigit, file_name)))

    # Iterate over each folder
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        
        # Get the list of images, sorted numerically
        images = sorted(os.listdir(folder_path), key=numeric_sort)[:num_images]
        
        # Display the images
        plt.figure(figsize=(15, 5))
        
        for idx, img_name in enumerate(images):
            img_path = os.path.join(folder_path, img_name)
            img = mpimg.imread(img_path)
            plt.subplot(2, 5, idx+1)  # Adjust layout to show 10 images (2 rows and 5 columns)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'{img_name}')
        
        plt.savefig(f"src/EDA/Images/{folder}_images.png")
        plt.close()

def findDistribution(folders, base_dir):
    for folder in folders:
        folder_path = f"{base_dir}/{folder}"
        files = os.listdir(folder_path)
        print(f"{folder} length: {len(files)}")


if __name__ == "__main__":
    folders = ['mask_weared_incorrect', 'with_mask', 'without_mask']
    base_dir = "C:/Users/ndonato/Downloads/archive/sorted_images_cropped"
    
    preview_images(folders, base_dir)
    findDistribution(folders, base_dir)


