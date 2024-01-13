import os
import shutil, random

# Specify the number of files to copy from each subfolder
num_files_to_copy_per_subfolder = 500

source_dir = './DATA-stl10/train'  # Replace with the path to your source directory
destination_dir = f'./DATA-stl10_sub_{num_files_to_copy_per_subfolder}/train'  # Replace with the path to your destination directory

if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)


for root, dirs, files in os.walk(source_dir):
    for subfolder in dirs:
        source_subfolder = os.path.join(root, subfolder)
        destination_subfolder = os.path.join(destination_dir, subfolder)
        
        # Create the corresponding subfolder in the destination directory
        if not os.path.exists(destination_subfolder):
            os.makedirs(destination_subfolder)

        # List all items (files and subfolders) in the source subfolder
        all_items = os.listdir(source_subfolder)

        # Select num_files_to_copy_per_subfolder random items (you can change this number as needed)
        selected_items = random.sample(all_items, num_files_to_copy_per_subfolder)

        # Copy the selected items to the destination subfolder
        for item in selected_items:
            source_item_path = os.path.join(source_subfolder, item)
            destination_item_path = os.path.join(destination_subfolder, item)
            if os.path.isfile(source_item_path):
                shutil.copy2(source_item_path, destination_item_path)  # Copy files
            elif os.path.isdir(source_item_path):
                shutil.copytree(source_item_path, destination_item_path)  # Copy subfolders and their contents
