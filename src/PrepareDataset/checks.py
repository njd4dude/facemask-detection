def check_dataset(dataset):
    print(f"size: {len(dataset)}")

    image, label = dataset[0]
    print(f"Sample image shape: {image.shape}")
    print(f"Sample label: {label}")

def check_dataloader(dataloader):
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx+1}: Input shape: {inputs.shape}, Labels: {labels.shape}")
        break  # Only process one batch for debugging
