import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
base_dir = "C:\s7 60 rework\output_slices(sagittal)"  # Current directory containing images in category folders
newdir="trainval(sagittal)"
train_dir = os.path.join(newdir, "train")
valid_dir = os.path.join(newdir, "valid")

# Create train and valid directories
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(valid_dir):
    os.makedirs(valid_dir)

# Categories (subfolders) inside base_dir
categories = ["displaced_rib_fracture", "non_displaced_rib_fracture", "buckle_rib_fracture", "segmental_rib_fracture", "no_fractures"]

# Split each category into train and valid
for category in categories:
    category_path = os.path.join(base_dir, category)
    if not os.path.exists(category_path):
        print(f"Category folder {category} does not exist, skipping...")
        continue

    # List all files in the category folder
    images = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
    
    # Split into train and valid sets (80% train, 20% valid)
    train_files, valid_files = train_test_split(images, test_size=0.2, random_state=42)

    # Create category subfolders inside train and valid directories
    train_category_path = os.path.join(train_dir, category)
    valid_category_path = os.path.join(valid_dir, category)
    os.makedirs(train_category_path, exist_ok=True)
    os.makedirs(valid_category_path, exist_ok=True)

    # Move files to train folder
    for file in train_files:
        src = os.path.join(category_path, file)
        dst = os.path.join(train_category_path, file)
        shutil.move(src, dst)

    # Move files to valid folder
    for file in valid_files:
        src = os.path.join(category_path, file)
        dst = os.path.join(valid_category_path, file)
        shutil.move(src, dst)

    print(f"Processed category: {category}")

print("Dataset split into train and valid folders successfully!")

#hello
