import os
import numpy as np
from PIL import Image
import h5py

def load_images_from_folder(folder, image_size):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder, filename)
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')  # Ensure 3 channels
                    img = img.resize((image_size, image_size))
                    img_array = np.array(img)
                    images.append(img_array)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    return images

def save_dataset_to_h5(X, y, classes, filename):
    with h5py.File(filename, 'w') as h5file:
        h5file.create_dataset('X', data=X)
        h5file.create_dataset('y', data=y)
        h5file.create_dataset('classes', data=classes)

def create_datasets():
    image_size = 64

    # Paths to image directories
    train_cats_dir = 'dataset/train/cats'
    train_non_cats_dir = 'dataset/train/non_cats'
    test_cats_dir = 'dataset/test/cats'
    test_non_cats_dir = 'dataset/test/non_cats'

    # Load images
    train_cats = load_images_from_folder(train_cats_dir, image_size)
    train_non_cats = load_images_from_folder(train_non_cats_dir, image_size)
    test_cats = load_images_from_folder(test_cats_dir, image_size)
    test_non_cats = load_images_from_folder(test_non_cats_dir, image_size)

    # Combine images and labels
    train_images = train_cats + train_non_cats
    train_labels = [1]*len(train_cats) + [0]*len(train_non_cats)
    test_images = test_cats + test_non_cats
    test_labels = [1]*len(test_cats) + [0]*len(test_non_cats)

    # Convert to NumPy arrays
    train_set_x_orig = np.array(train_images)
    train_set_y_orig = np.array(train_labels)
    test_set_x_orig = np.array(test_images)
    test_set_y_orig = np.array(test_labels)

    # Shuffle datasets
    train_perm = np.random.permutation(len(train_set_y_orig))
    train_set_x_orig = train_set_x_orig[train_perm]
    train_set_y_orig = train_set_y_orig[train_perm]

    test_perm = np.random.permutation(len(test_set_y_orig))
    test_set_x_orig = test_set_x_orig[test_perm]
    test_set_y_orig = test_set_y_orig[test_perm]

    # Define class labels
    classes = np.array([b'non-cat', b'cat'])

    # Create datasets directory if it doesn't exist
    if not os.path.exists('datasets'):
        os.makedirs('datasets')

    # Save datasets to H5 files
    save_dataset_to_h5(train_set_x_orig, train_set_y_orig, classes, 'datasets/train_catvnoncat.h5')
    save_dataset_to_h5(test_set_x_orig, test_set_y_orig, classes, 'datasets/test_catvnoncat.h5')

# Run the function to create datasets
create_datasets()
