import os
import shutil
import random

# Paths
original_dataset_dir = r"D:\2D_to_3D\TB_Chest_Radiography_Database"  # Your original dataset
base_dir = r"D:\2D_to_3D\dataset"        # Output base dir

# Split ratio
split_ratio = 0.8  # 80% train, 20% val

# Ensure base directory is fresh
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
os.makedirs(base_dir)

# Create train/val folders
for category in ['train', 'val']:
    for class_name in ['Normal', 'Tuberculosis']:
        os.makedirs(os.path.join(base_dir, category, class_name))

# Copy files into train/val folders
for class_name in ['Normal', 'Tuberculosis']:
    class_dir = os.path.join(original_dataset_dir, class_name)
    images = os.listdir(class_dir)
    random.shuffle(images)

    split_point = int(len(images) * split_ratio)
    train_images = images[:split_point]
    val_images = images[split_point:]

    # Copy to train
    for img in train_images:
        src = os.path.join(class_dir, img)
        dst = os.path.join(base_dir, 'train', class_name, img)
        shutil.copy2(src, dst)

    # Copy to val
    for img in val_images:
        src = os.path.join(class_dir, img)
        dst = os.path.join(base_dir, 'val', class_name, img)
        shutil.copy2(src, dst)

print("✅ Dataset successfully split into train/val!")
