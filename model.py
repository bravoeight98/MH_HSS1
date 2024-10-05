import tensorflow as tf
from tensorflow.keras import layers, models
import os
import cv2
import numpy as np

# Define the CNN model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(5, activation='softmax')  # 5 face shapes
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def load_data(data_dir):
    labels = {'heart': 0, 'long': 1, 'oval': 2, 'round': 3, 'square': 4}
    data = []
    target = []
    for label, idx in labels.items():
        folder_path = os.path.join(data_dir, label)
        for image_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, image_name)
            image = cv2.imread(img_path)
            image = cv2.resize(image, (128, 128))
            data.append(image)
            target.append(idx)
    return np.array(data), np.array(target)

# Train and save the model
if __name__ == "__main__":
    data_dir = "E:\\_MH\\MH_HSS1\\dataset\\"
    X, y = load_data(data_dir)
    model = create_model()
    model.fit(X, y, epochs=10)
    model.save('face_shape_model.h5')
