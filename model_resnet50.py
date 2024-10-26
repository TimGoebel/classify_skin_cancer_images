import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def create_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def plot_and_save_history(history, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'training_history_ResNet50.png'))
    plt.close()

def train_resnet(new_directory_path, directory):
    model_dir = os.path.join(directory, 'models')
    os.makedirs(model_dir, exist_ok=True)
    plot_dir = os.path.join(directory, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Load preprocessed data
    X_train = np.load(os.path.join(new_directory_path, 'X_train.npy'))
    X_val = np.load(os.path.join(new_directory_path, 'X_val.npy'))
    y_train = np.load(os.path.join(new_directory_path, 'y_train.npy'))
    y_val = np.load(os.path.join(new_directory_path, 'y_val.npy'))

    # Check for NaN or Infinite values
    print("***** Check for NaN or Infinite Values:")
    print("X_train NaN:", np.isnan(X_train).sum(), "X_train Inf:", np.isinf(X_train).sum())
    print("y_train NaN:", np.isnan(y_train).sum(), "y_train Inf:", np.isinf(y_train).sum())

    # Create model
    model = create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='binary_focal_loss', metrics=['accuracy'])

    # Set up callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(os.path.join(model_dir, 'best_ResNet50_model.keras'), save_best_only=True)

    # Define data augmentation for training data with advanced techniques
    train_datagen = ImageDataGenerator(
        rescale=1./255,               # Normalize pixel values to [0, 1]
        rotation_range=40,            # Random rotations
        width_shift_range=0.2,        # Random horizontal shifts
        height_shift_range=0.2,       # Random vertical shifts
        shear_range=0.2,              # Random shearing
        zoom_range=0.2,               # Random zoom
        horizontal_flip=True,         # Random horizontal flip
        # vertical_flip=True,           # Random vertical flip
        fill_mode='nearest'          # Filling strategy for newly created pixels
        # brightness_range=[0.8, 1.2]  # Random brightness adjustments
        # channel_shift_range=0.2       # Random channel shifts
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    # Create data generators
    train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=32)

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20,
        callbacks=[early_stopping, model_checkpoint]
    )

    # Save training history plot
    plot_and_save_history(history, plot_dir)

    # Unfreeze some of the deeper layers for fine-tuning
    for layer in model.layers[-10:]:
        layer.trainable = True

    optimizer_fine = tf.keras.optimizers.Adam(learning_rate=1e-7)
    model.compile(optimizer=optimizer_fine, loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=optimizer_fine, loss='binary_focal_loss', metrics=['accuracy'])

    # Continue training with fine-tuning
    history_fine = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
        callbacks=[early_stopping, model_checkpoint]
    )

    # Save fine-tuning history plot
    plot_and_save_history(history_fine, plot_dir)

    # Save the final model
    model.save(model_dir + '/resnet50_final_model.h5')