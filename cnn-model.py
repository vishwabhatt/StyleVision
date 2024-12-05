"""
!pip install tensorflow
!pip install scikit-learn
!pip install pandas
!pip install matplotlib
!pip install seaborn
!pip install keras-vis
"""
!pip install keras-vis --upgrade

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder  
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras import backend as K
from collections import abc

import seaborn as sns
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
import os
import numpy as np

# PreStep 1: Load Data from Zip File
def load_data_from_zip(zip_file_path="FashionProductImages-Original.zip", image_folder_name="images", csv_file_name="styles.csv"):
  """
  Extracts data (images and labels) from a zip file.

  Args:
      zip_file_path: Path to the zip file containing the data.
      image_folder_name: Name of the folder containing images inside the zip (default: "images").
      csv_file_name: Name of the CSV file containing labels (default: "styles.csv").

  Returns:
      A tuple containing the loaded dataframe and the extracted image directory.
  """
  with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall()
    extracted_dir = os.path.splitext(zip_file_path)[0]  # Extract directory name
    df = pd.read_csv(csv_file_name)
    return df, os.path.join(extracted_dir, image_folder_name)

# Load data from the zip file
zip_file_path = "FashionProductImages-Original.zip"
df, image_dir = load_data_from_zip(zip_file_path)

# Step 1: Load the CSV File
df = pd.read_csv('styles.csv')
image_folder_name = "images"

# Convert 'id' column to string
df['id'] = df['id'].astype(str)

# Encode labels using LabelEncoder
label_encoder = LabelEncoder()
df['articleType_encoded'] = label_encoder.fit_transform(df['articleType'])

# Get the number of classes
num_classes = len(label_encoder.classes_)

# Split data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Function to display one image per class
def display_one_image_per_class(df, image_folder_name):
    """Displays one image per class from the DataFrame."""
    unique_classes = df['articleType'].unique()

    plt.figure(figsize=(20, 20))
    for i, class_name in enumerate(unique_classes):
        # Find the first image of the current class
        image_id = df[df['articleType'] == class_name]['id'].iloc[0]
        image_path = os.path.join(image_folder_name, str(image_id) + '.jpg')

        # Load and display the image
        img = load_img(image_path, target_size=(256, 256))
        plt.subplot(len(unique_classes) // 5 + 1, 5, i + 1)  # Adjust grid as needed
        plt.imshow(img)
        plt.title(class_name)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Call the function to display images
display_one_image_per_class(df, image_folder_name) 

"""
# Step 2: Create a Custom Data Generator
def custom_data_generator(df, batch_size=32, img_size=(128, 128)):
    while True:
        X_batch = []
        y_batch = []
        for _, row in df.sample(n=batch_size).iterrows():
            img_path = os.path.join(image_folder_name, row['id'] + '.jpg')  # Assuming image names are 'id' + '.jpg'
            #img = load_img(img_path, target_size=img_size)
            #img_array = img_to_array(img) / 255.0
            #X_batch.append(img_array)
            #y_batch.append(row['articleType_encoded'])  # Use encoded labels
            if os.path.exists(img_path):
                img = load_img(img_path, target_size=img_size)
                img_array = img_to_array(img) / 255.0
                X_batch.append(img_array)
                y_batch.append(row['articleType_encoded'])  # Use encoded labels
            else:
                print(f"Warning: Image file not found: {img_path}")  # Print a warning

        X_batch = np.array(X_batch)
        y_batch = to_categorical(y_batch, num_classes=len(label_encoder.classes_))  # One-hot encode labels
        yield X_batch, y_batch
"""
"""
# Step 2: Create a Custom Data Generator
def custom_data_generator(df, batch_size=32, img_size=(128, 128)):
    # Get the total number of samples in the DataFrame
    num_samples = len(df)
    
    # Initialize the index for iteration
    i = 0
    
    # Loop until the index 'i' reaches the total number of samples
    while True:  # Changed to while True to ensure continuous iteration
        X_batch = []
        y_batch = []
        
        # Iterate through a batch of samples
        for j in range(batch_size):
            # Calculate the current index within the DataFrame, handling edge cases
            current_index = (i + j) % num_samples  
            
            # Get the data row for the current index
            row = df.iloc[current_index]
            
            img_path = os.path.join(image_folder_name, row['id'] + '.jpg')
            
            if os.path.exists(img_path):
                img = load_img(img_path, target_size=img_size)
                img_array = img_to_array(img) / 255.0
                X_batch.append(img_array)
                y_batch.append(row['articleType_encoded'])
            else:
                print(f"Warning: Image file not found: {img_path}")
        
        # Convert lists to NumPy arrays
        X_batch = np.array(X_batch)
        y_batch = to_categorical(y_batch, num_classes=len(label_encoder.classes_))
        
        # Yield the batch of data
        yield X_batch, y_batch
        
        # Update the index for the next iteration
        i += batch_size
"""
# Step 2: Create a Custom Data Generator
def custom_data_generator(df, batch_size=32, img_size=(128, 128), shuffle=True):  # Added shuffle argument
    # Get the total number of samples in the DataFrame
    num_samples = len(df)
    
    # Initialize the index for iteration
    i = 0
    
    # Loop until the index 'i' reaches the total number of samples
    while True:  # Changed to while True to ensure continuous iteration
        X_batch = []
        y_batch = []
        
        # Iterate through a batch of samples
        # If shuffle is True, shuffle the DataFrame before getting the batch
        if shuffle:
            batch_df = df.sample(n=batch_size)  # Shuffle the DataFrame
        else:
            batch_df = df.iloc[i:i + batch_size]  # Get a sequential batch
        
        for _, row in batch_df.iterrows():  # Iterate through the batch DataFrame
            img_path = os.path.join(image_folder_name, row['id'] + '.jpg')
            
            if os.path.exists(img_path):
                img = load_img(img_path, target_size=img_size)
                img_array = img_to_array(img) / 255.0
                X_batch.append(img_array)
                y_batch.append(row['articleType_encoded'])
            else:
                print(f"Warning: Image file not found: {img_path}")
        
        # Convert lists to NumPy arrays
        X_batch = np.array(X_batch)
        y_batch = to_categorical(y_batch, num_classes=len(label_encoder.classes_))
        
        # Yield the batch of data
        yield X_batch, y_batch
        
        # Update the index for the next iteration (only if shuffle is False)
        if not shuffle:
            i += batch_size
            if i >= num_samples:  # Reset index if it exceeds the DataFrame length
                i = 0

# Create data generators
train_generator = custom_data_generator(train_df)
val_generator = custom_data_generator(val_df)


# Create the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128,  
 128, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),  

    Dropout(0.5),  

    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,  
  # Adjust epochs as needed
    steps_per_epoch=len(train_df) // 32, # Assuming batch_size is 32
    validation_steps=len(val_df) // 32
)
"""
# Plot training history
plt.figure(figsize=(12, 4))  # Adjust figure size as needed

# Plot accuracy
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Plot loss
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
"""
# Evaluate the model
# Reset the generators before evaluation
val_generator = custom_data_generator(val_df, batch_size=32)  # Resetting val_generator

# Calculate validation steps based on actual data size and batch size
validation_steps = (len(val_df) + 32 - 1) // 32  # Ensure all data is processed in validation

test_loss, test_acc = model.evaluate(val_generator, steps=validation_steps)
print('Test accuracy:', test_acc)

"""
test_loss, test_acc = model.evaluate(val_generator)
print('Test accuracy:', test_acc)  
"""

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')  

plt.title('Accuracy')
plt.legend()
plt.show()  


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.legend()
plt.show()  


# Save the model
model.save('fashion_model.h5')

"""
# Function to predict a class for a new image
def predict_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)  

    predicted_class_label = list(train_generator.class_indices.keys())[predicted_class_index]
    print("Predicted class:", predicted_class_label)

# Example usage:
test_image_path = "images/1164.jpg"
predict_image(test_image_path)
"""

# Confusion Matrix
# Get predictions on the validation set
val_generator = custom_data_generator(val_df, batch_size=32, shuffle=False)  # Do not shuffle for confusion matrix
y_pred = model.predict(val_generator, steps=validation_steps)
y_pred_classes = np.argmax(y_pred, axis=1)  # Get predicted class labels
y_true = val_df['articleType_encoded'].values[:len(y_pred_classes)]  # Get true class labels

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(15, 15))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Function to predict a class for a new image
def predict_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)  

    # Use label_encoder.inverse_transform to get the original label
    predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]  
    print("Predicted class:", predicted_class_label)

# Example usage:
test_image_path = "images/1164.jpg"
predict_image(test_image_path)
