import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TF_ENABLE_ONEDNN_OPTS=0

def adjust_contrast(img, level=1.5):
    return cv2.convertScaleAbs(img, alpha=level, beta=0)

def adjust_saturation(img, level=1.5):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img[:, :, 1] = cv2.multiply(hsv_img[:, :, 1], level)
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

def adjust_brightness(img, level=30):
    return cv2.convertScaleAbs(img, alpha=1, beta=level)

def add_noise(img, level=0.05):
    noise = np.random.randn(*img.shape) * 255 * level
    noisy_img = img + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def adjust_exposure(img, level=0.5):
    return cv2.convertScaleAbs(img, alpha=level, beta=0)

def save_augmented_image(img, img_path, suffix):
    base, ext = os.path.splitext(img_path)
    new_img_path = f"{base}_{suffix}{ext}"
    cv2.imwrite(new_img_path, img)
    return new_img_path

def relabel_files_in_directory(directory, data_dir, category):
    save_dir = os.path.join(data_dir, f"{category}_augmented")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    files = sorted(os.listdir(directory))
    for i, filename in enumerate(files, start=1):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            new_file_name = f"{i}.jpg"
            new_file_path = os.path.join(save_dir, new_file_name)
            while os.path.exists(new_file_path):
                i += 1
                new_file_name = f"{i}.jpg"
                new_file_path = os.path.join(save_dir, new_file_name)
            os.rename(file_path, new_file_path)
    return save_dir

def augmentations(data_dir):
    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            
            # Check if the image file exists before trying to read it
            if not os.path.exists(img_path):
                print(f"Warning: File {img_path} does not exist.")
                continue

            img = cv2.imread(img_path)
            
            # Ensure that the image is successfully loaded
            if img is None:
                print(f"Error: Could not read the image {img_path}. It may be corrupted or not a valid image file.")
                continue

            # Augmentations
            augmented_images = [
                ('contrast', adjust_contrast(img)),
                ('saturation', adjust_saturation(img)),
                ('brightness', adjust_brightness(img)),
                ('noise', add_noise(img)),
                ('exposure', adjust_exposure(img))
            ]
            
            for suffix, aug_img in augmented_images:
                save_augmented_image(aug_img, img_path, suffix)

        relabel_files_in_directory(category_path, data_dir, category)
    

def load_images(data_dir, img_size):
    images = []
    labels = []
    for category in ['Benign_augmented', 'Malignant_augmented']:
        category_path = os.path.join(data_dir, category)
        label = 1 if 'Malignant_augmented' in category else 0
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:  # Ensure the image is loaded successfully
                img = cv2.resize(img, (img_size, img_size))
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

def preprocess_main(data_dir, img_size):
    augmentations(data_dir)
    # Load and preprocess data
    print("Load and augment images")
    images, labels = load_images(data_dir, img_size)
    # Implement further data preprocessing here
    print("Split the data")
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

    print("Filter for .npy files")
    npy_files = [f for f in data_dir if f.endswith('.npy')]

    # Preprocessed data files found
    if npy_files:
        print("Found .npy files:")
    # Save preprocessed data if necessary
    else:
        print("No .npy files found in the directory.")
        np.save(data_dir+'/X_train.npy', X_train)
        np.save(data_dir+'/X_val.npy', X_val)
        np.save(data_dir+'/X_test', X_test)
        np.save(data_dir+'/y_train.npy', y_train)
        np.save(data_dir+'/y_val.npy', y_val)
        np.save(data_dir+'/y_test.npy', y_test)


