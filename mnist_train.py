import argparse
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import Callback
import neptune
import numpy as np

# Parse the following arguments from the terminal:
# --num-epochs      The number of epochs, default is 10
# --learning_rate   The learning rate, default 0.001
# --run_name        The name of the run
parser = argparse.ArgumentParser(description='Train a CNN model on MNIST dataset')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
parser.add_argument('--run_name', type=str, default='my_first_experiment', help='Name of the experiment run')
args = parser.parse_args()

# Set TensorFlow to run on CPU
tf.config.set_visible_devices([], 'GPU')

# Name of the experiment
run_name = args.run_name

run = neptune.init_run(
    project="emma.saroyan/Mnist",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzODFkZTZkMi05M2FiLTQ2YzctYjQxOS1iMTIyNmFmNzNhNDQifQ==",
    name=run_name
)  # your credential

class TrainingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Logs/metrics for each epoch
        run['loss'].append(logs['loss'])
        run['accuracy'].append(logs['accuracy'])
        run['val_loss'].append(logs['val_loss'])
        run['val_accuracy'].append(logs['val_accuracy'])

        print(f"Epoch {epoch}: Loss: {logs['loss']}, Accuracy: {logs['accuracy']}, Validation Loss: {logs['val_loss']}, Validation Accuracy: {logs['val_accuracy']}")

# Load and preprocess data
(x_train, y_train), (x_val, y_val) = mnist.load_data()

# Print out some statistics for the dataset
total_images = x_train.shape[0] + x_val.shape[0]
print(f"Total number of images in the dataset: {total_images}")
print(f"Number of training images: {x_train.shape[0]}")
print(f"Number of validation images: {x_val.shape[0]}")
print(f"Training set percentage: {100 * x_train.shape[0] / total_images:.2f}%")
print(f"Validation set percentage: {100 * x_val.shape[0] / total_images:.2f}%")

# Class distribution
unique, counts = np.unique(y_train, return_counts=True)
train_class_distribution = dict(zip(unique, counts))
unique, counts = np.unique(y_val, return_counts=True)
val_class_distribution = dict(zip(unique, counts))

# Metadata about the dataset
run['dataset'] = {
    "total_images": total_images,
    "total_train": x_train.shape[0],
    "total_val": x_val.shape[0],
    "train_percentage": 100 * x_train.shape[0] / total_images,
    "val_percentage": 100 * x_val.shape[0] / total_images
}

print("Training set class distribution and percentages:")
for k, v in train_class_distribution.items():
    print(f"Class {k}: {v} images, {100 * v / x_train.shape[0]:.2f}%")

print("Validation set class distribution and percentages:")
for k, v in val_class_distribution.items():
    print(f"Class {k}: {v} images, {100 * v / x_val.shape[0]:.2f}%")



x_train, x_val = x_train / 255.0, x_val / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_val = x_val.reshape(-1, 28, 28, 1)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_val = tf.keras.utils.to_categorical(y_val, 10)

# Define model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Configure the learning rate
learning_rate = args.learning_rate
# Configure the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Model Parameters
num_epochs = args.num_epochs
run['parameters'] = {
    "num_epochs": num_epochs,
    "optimizer": optimizer.name,
    "learning_rate": learning_rate
}

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=num_epochs, callbacks=[TrainingCallback()])

run.stop()