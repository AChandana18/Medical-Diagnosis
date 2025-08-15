import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from xai.grad_cam import generate_grad_cam 

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = 'dataset'

# Data generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Checkpoint to save best model
checkpoint = ModelCheckpoint('best_medical_model.h5', save_best_only=True, monitor='val_loss', mode='min')

# Train
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[checkpoint]
)

# Evaluate
loss, acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {acc:.4f}")

# Load best model

model = load_model('best_medical_model.h5')

# Predict on a few samples and apply Grad-CAM
sample_images, sample_labels = next(val_generator)
predictions = model.predict(sample_images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(sample_labels, axis=1)
class_labels = list(train_generator.class_indices.keys())

for i in range(5):
    plt.imshow(sample_images[i])
    plt.title(f"Predicted: {class_labels[predicted_classes[i]]}, Actual: {class_labels[true_classes[i]]}")
    plt.axis('off')
    plt.show()

    # Grad-CAM visualization
    heatmap = generate_grad_cam(model, sample_images[i], predicted_classes[i])
    plt.imshow(sample_images[i])
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title("Grad-CAM")
    plt.axis('off')
    plt.show()
