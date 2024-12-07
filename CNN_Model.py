import os
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Step 1: Extract and Load Dataset
extract_dir = "/content/dataset"  # Extraction directory

# Step 2: Preprocess Images
IMG_HEIGHT, IMG_WIDTH = 128, 128  # Reduced image size
processed_images = []
labels = []

for root, _, files in os.walk(extract_dir):
    for file in files:
        img_path = os.path.join(root, file)
        try:
            # Load and preprocess images
            img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode="rgb")
            img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
            processed_images.append(img_array)
            labels.append(os.path.basename(root))  # Use folder name as label
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Convert to numpy arrays
X = np.array(processed_images)
y = np.array(labels)
print(f"Processed {len(X)} images of shape {X.shape}")

# Step 3: Encode Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)
print(f"Classes: {label_encoder.classes_}")

# Step 4: Split Dataset
X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# Step 5: Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# Step 6: Build Transfer Learning Model with MobileNet
base_model = MobileNet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)  # Reduced dense layer size
x = Dropout(0.4)(x)
outputs = Dense(len(label_encoder.classes_), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Step 7: Callbacks for Early Stopping and Learning Rate Reduction
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Step 8: Train the Model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=16),  # Reduced batch size
    validation_data=(X_val, y_val),
    epochs=20,  # Reduced epochs
    callbacks=[early_stopping, reduce_lr]
)

# Step 9: Evaluate the Model
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2%}")

# Step 10: Confusion Matrix and Classification Report
y_val_pred = model.predict(X_val)
y_val_pred_classes = np.argmax(y_val_pred, axis=1)
y_val_true_classes = np.argmax(y_val, axis=1)

conf_matrix = confusion_matrix(y_val_true_classes, y_val_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

print("\nClassification Report:")
print(classification_report(y_val_true_classes, y_val_pred_classes, target_names=label_encoder.classes_))

# Step 11: Save Model
model.save('fashion_cnn_optimized_model.h5')
print("Model saved as fashion_cnn_optimized_model.h5") 