import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import os

# Step 1: Dataset Paths
dataset_dir = "/content/fashion-dataset"  # Update with the dataset folder path
test_image_path = "/content/fashion-dataset/images/10000.jpg"  # Update with a test image path

# Step 2: Data Preprocessing
datagen = ImageDataGenerator(
    rescale=1.0 / 255,  
    validation_split=0.2  
)

train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(128, 128),  
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# Step 3: Define CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
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
    Dense(train_generator.num_classes, activation='softmax')  
])

# Step 4: Compile the Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train the Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,  
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)

# Step 6: Evaluate the Model
val_loss, val_acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {val_acc:.2f}")

# Step 7: Save the Model
model.save("fashion_cnn_model.h5")
print("Model saved as 'fashion_cnn_model.h5'.")

# Step 8: Plot Training Progress
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Step 9: Test with a New Image
def predict_image(image_path):
    img = load_img(image_path, target_size=(128, 128))  
    img_array = img_to_array(img) / 255.0 
    img_array = np.expand_dims(img_array, axis=0)  

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_labels = list(train_generator.class_indices.keys())  
    print(f"Predicted Class: {class_labels[predicted_class]}")

# Predict a test image
predict_image(test_image_path)

# Step 10: Classification Report (Optional)
y_true = val_generator.classes
y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
class_labels = list(val_generator.class_indices.keys())
print(classification_report(y_true, y_pred_classes, target_names=class_labels))
