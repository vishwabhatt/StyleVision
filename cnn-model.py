import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator  
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import pandas as pd
import zipfile
import matplotlib.pyplot as plt
import os

# Step 1: Load the CSV File
df = pd.read_csv('your_csv_file.csv')

# Step 2: Create a Data Generator
def generate_data(df, batch_size=32, img_size=(128, 128)):
    while True:
        X_batch = []
        y_batch = []
        for _, row in df.sample(n=batch_size).iterrows():
            img_path = os.path.join('images_folder', row['image_filename'])
            img = load_img(img_path, target_size=img_size)
            img_array = img_to_array(img) / 255.0
            X_batch.append(img_array)
            y_batch.append(row['label'])  # Assuming 'label' is the column name for class labels

        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)

        yield X_batch, y_batch

# Create a data generator
train_generator = generate_data(df)

# Create an ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)

# Convert 'id' column to string
df['id'] = df['id'].astype(str)

# Create a data generator from the DataFrame --------------------->>>>>>>>>>>>>>>>>>>Working on error that occurs below
train_generator = train_datagen.flow_from_dataframe(
    dataframe=df,
    directory='extracted_images/images',
    x_col='id',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2,  
 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  

    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
