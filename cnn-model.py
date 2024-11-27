import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator  

import pandas as pd
import zipfile

# Unzip the images
with zipfile.ZipFile('FashionProductImages.zip', 'r') as zip_ref:
    zip_ref.extractall('extracted_images')

# Load the CSV file
df = pd.read_csv('styles.csv')

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
