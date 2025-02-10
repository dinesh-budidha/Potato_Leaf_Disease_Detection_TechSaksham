import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# Configuration
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50
DATASET_PATH = "PlantVillage"
MODEL_FILENAME = 'potato_model.sav'

# Load dataset
def load_dataset(path, image_size, batch_size):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        shuffle=True,
        image_size=(image_size, image_size),
        batch_size=batch_size
    )
    class_names = dataset.class_names
    print(f"Class names: {class_names}")
    return dataset, class_names

# Visualize sample images
def visualize_samples(dataset, class_names):
    plt.figure(figsize=(10, 10))
    for image_batch, label_batch in dataset.take(1):
        for i in range(12):
            ax = plt.subplot(3, 4, i + 1)
            plt.imshow(image_batch[i].numpy().astype("uint8"))
            plt.title(class_names[label_batch[i]])
            plt.axis("off")
    plt.show()

# Split dataset into train, validation, and test sets
def split_dataset(dataset, train_split=0.8, val_split=0.1):
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    
    train_ds = dataset.take(train_size)
    test_ds = dataset.skip(train_size)
    val_ds = test_ds.take(val_size)
    test_ds = test_ds.skip(val_size)
    
    return train_ds, val_ds, test_ds

# Prepare datasets for training
def prepare_datasets(train_ds, val_ds, test_ds):
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_ds, val_ds, test_ds

# Define CNN model
def create_model(input_shape, num_classes):
    model = models.Sequential([
        resize_and_rescale,
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.summary()
    return model

# Plot training history
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(EPOCHS), acc, label='Training Accuracy')
    plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(EPOCHS), loss, label='Training Loss')
    plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.show()

# Predict and visualize results
def predict_and_visualize(model, test_ds, class_names):
    for images, labels in test_ds.take(1):
        plt.figure(figsize=(15, 15))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            
            img_array = tf.expand_dims(images[i].numpy(), axis=0)
            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = round(100 * np.max(predictions[0]), 2)
            
            actual_class = class_names[labels[i]]
            plt.title(f"Actual: {actual_class}\nPredicted: {predicted_class}\nConfidence: {confidence}%")
            plt.axis("off")
        plt.show()

# Main script
if __name__ == "__main__":
    # Load and visualize the dataset
    dataset, class_names = load_dataset(DATASET_PATH, IMAGE_SIZE, BATCH_SIZE)
    visualize_samples(dataset, class_names)
    
    # Split and prepare datasets
    train_ds, val_ds, test_ds = split_dataset(dataset)
    train_ds, val_ds, test_ds = prepare_datasets(train_ds, val_ds, test_ds)
    
    # Define preprocessing and augmentation layers
    resize_and_rescale = tf.keras.Sequential([
        layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        layers.Rescaling(1. / 255)
    ])
    
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2)
    ])
    
    # Create and train the model
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    model = create_model(input_shape, len(class_names))
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    
    # Evaluate the model
    scores = model.evaluate(test_ds)
    print(f"Test Accuracy: {scores[1] * 100:.2f}%")
    
    # Save the trained model
    pickle.dump(model, open(MODEL_FILENAME, 'wb'))
    
    # Plot training history
    plot_training_history(history)
    
    # Predict and visualize test results
    predict_and_visualize(model, test_ds, class_names)
