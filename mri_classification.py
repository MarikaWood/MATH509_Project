## STEP 1: Install & Configure Kaggle CLI
import os
from google.colab import files

!pip install -q kaggle

# Tell Kaggle CLI to look for the API key in /content
os.environ["KAGGLE_CONFIG_DIR"] = "/content"

# Manually upload kaggle.json file
uploaded = files.upload()  # select kaggle.json when prompted

# Create a ~/.kaggle folder
!mkdir -p ~/.kaggle

# Move kaggle.json into ~/.kaggle folder
!cp /content/kaggle.json ~/.kaggle/
!chmod 600 /content/kaggle.json



## STEP 2: Download & Unzip the Dataset

# Download the Images Oasis dataset from Kaggle
!kaggle datasets download ninadaithal/imagesoasis -p /content --force

# Unzip into /content/Data
!unzip -o /content/imagesoasis.zip -d /content

# Check extracted folder name
print("Extracted files:", os.listdir("/content/Data"))

# Expected structure:
# /content/Data/
#   Mild Dementia/
#   Moderate Dementia/
#   Non Demented/
#   Very Mild Dementia/

# Each folder has a set of slices for various patients, e.g.:
#   OAS1_0285_MR1_mpr-1_128.jpg (0285 = patient, MR1 = 1st MRI, 128 = MRI slice)



## STEP 3: Filter to Use Slices 119/120/121 Per Patient
import shutil
import re
import os
import random
import numpy as np

# Define root directory where dataset is stored
data_root = "/content/Data"

# Define the directory for filtered images
filtered_root = "/content/Data_Selected_Slices"
if os.path.exists(filtered_root):
    shutil.rmtree(filtered_root)  # Clear existing filtered data
os.makedirs(filtered_root, exist_ok=True)

# Define dementia classes (excluding Moderate Dementia)
classes = ["Mild Dementia", "Non Demented", "Very mild Dementia"]

# Regex to extract patient ID and slice index from filenames
pattern = re.compile(r"(OAS1_\d+)_MR\d+_mpr-\d+_(\d+)\.jpg")

# Dictionary to track patients and slices
patient_slices = {cls: {} for cls in classes}

for cls in classes:
    class_in_path = os.path.join(data_root, cls)

    for fname in os.listdir(class_in_path):
        match = pattern.match(fname)
        if match:
            patient_id = match.group(1)   # Extract patient ID
            slice_idx = int(match.group(2))    # Extract slice number

            # Only keep slices 119, 120, 121
            if slice_idx in [119, 120, 121]:
                if patient_id not in patient_slices[cls]:
                    patient_slices[cls][patient_id] = []
                patient_slices[cls][patient_id].append(fname)

# Perform random sampling of 6 images per patient and copy to filtered directory
for cls, patients in patient_slices.items():
    class_out_path = os.path.join(filtered_root, cls)
    os.makedirs(class_out_path, exist_ok=True)

    for patient_id, images in patients.items():
        # Select 6 images per patient
        np.random.seed(88) # Set a random seed for reproducibility
        selected_images = random.sample(images, min(6, len(images)))

        # Copy only the selected images
        for img in selected_images:
            src = os.path.join(data_root, cls, img)
            dst = os.path.join(class_out_path, img)
            shutil.copy2(src, dst)

print("\nRandom sampling complete. 6 images per patient selected.\n")

# Count unique patients per class (use to check expected number of image per class)
print("\nUnique Patients per Class:")
for cls, patients in patient_slices.items():
    print(f"{cls}: {len(patients)} unique patients")

# Count images per class 
print("\nImage Count per Class After Selection:")
for cls in classes:
    class_path = os.path.join(filtered_root, cls)
    num_images = len(os.listdir(class_path))
    print(f"{cls}: {num_images} images")



## STEP 4: Split the Data into Train/Test (70:30)
import random

# Define directories
base_dir = "/content/OASIS_split"
train_dir = os.path.join(base_dir, "train")
test_dir  = os.path.join(base_dir, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Remove existing directories to prevent duplicate images when rerunning code
if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)

# First extract patient IDs per class
classes = ["Mild Dementia", "Non Demented", "Very mild Dementia"]
patient_dict = {cls: {} for cls in classes} # Store patients for each class

for cls in classes:
    cls_folder = os.path.join(filtered_root, cls)
    all_images = os.listdir(cls_folder)

    for fname in all_images:
        match = pattern.match(fname)
        if match:
            patient_id = match.group(1)  # Extract patient ID
            if patient_id not in patient_dict[cls]:
                patient_dict[cls][patient_id] = []
            patient_dict[cls][patient_id].append(fname)

# Split patients into train (70%) and test (30%) per class
random.seed(88) # Set seed for reproducible results
for cls, patient_images in patient_dict.items():
    patient_ids = list(patient_images.keys())
    random.shuffle(patient_ids)

    split_idx = int(0.7 * len(patient_ids))
    train_patients = patient_ids[:split_idx]
    test_patients  = patient_ids[split_idx:]

    # Make subfolders
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    # Copy the files to train set
    for patient_id in train_patients:
      for fname in patient_images[patient_id]:
        src = os.path.join(filtered_root, cls, fname)
        dst = os.path.join(train_dir, cls, fname)
        shutil.copy2(src, dst)

    # Copy the files to test set
    for patient_id in test_patients:
      for fname in patient_images[patient_id]:
        src = os.path.join(filtered_root, cls, fname)
        dst = os.path.join(test_dir, cls, fname)
        shutil.copy2(src, dst)

print("Data split complete.") # Patients are unique in train and test sets
print("Train folder:", train_dir)
print("Test folder:", test_dir)

# Count images per class in test and train sets
for folder in [train_dir, test_dir]:
    print(f"\nChecking: {folder}")
    for cls in os.listdir(folder):
        class_path = os.path.join(folder, cls)
        num_images = len(os.listdir(class_path))
        print(f"  {cls}: {num_images} images")

# Print unique patient count per class in train and test sets
print("\n===== Unique Patients per Class in Train/Test =====")
for cls in classes:
    total_patients = len(patient_dict[cls])  # Correct total count per class
    train_patients = int(0.7 * total_patients)  # 70% of patients
    test_patients = total_patients - train_patients  # Remaining 30%

    print(f"{cls}: {total_patients} total patients")
    print(f"  Train: {train_patients} patients")
    print(f"  Test: {test_patients} patients")



## STEP 5: Keras ImageDataGenerators with Class Balancing
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# Define image and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Training generator with augmentations
base_augmentation = dict(
    rescale=1.0/255,             # Pixel value normalization
    rotation_range=30,           # Randomly rotate by 30 deg
    width_shift_range=0.2,       # Shift width up to 20%
    height_shift_range=0.2,      # Shift height up to 20%
    zoom_range=0.3,              # Random zoom in/out by up to 30%
    horizontal_flip=True,        # Adds horizontally flipped versions of images
    brightness_range=[0.6, 1.4], # Adds variable brightness
    shear_range=15,              # Randomly slants the image up to 15 deg
    fill_mode='nearest'          # Fill gaps w/ nearest pixel value
)

# Create generators for training and testing
train_datagen = ImageDataGenerator(**base_augmentation)
test_datagen = ImageDataGenerator(rescale=1.0/255) # Test generator (only rescale, no aug)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True  # Make sure training data is shuffled
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # Keep order the same for consistent results
)

# Determine class weights
labels = train_generator.classes  # Integer labels
class_names = list(train_generator.class_indices.keys())  # Extract labels
class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

print("Class Weights:", class_weight_dict)
print("Class Indices:", train_generator.class_indices)



## STEP 6: Use MobileNetV2 (Transfer Learning)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# Load the MobileNetV2 model, pre-trained on ImageNet, excluding the top layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model layers initially

# Unfreeze last 3 layers for fine-tuning without overfitting
for layer in base_model.layers[-3:]:
    layer.trainable = True

# Build the model
model = models.Sequential([
    base_model,
    layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(0.01)),
    layers.BatchNormalization(),           # Helps with training stability
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),  # 256 neurons
    layers.Dropout(0.6),
    layers.Dense(3, activation='softmax')  # 3-class classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model summary
model.summary()



## STEP 7: Train the Model with Early Stopping & Class Weights
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np

# Early stopping (stop training if validation loss doesn't improve for 5 epochs)
early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)

# Reduce learning rate if validation loss plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
)

# Train the model
history = model.fit(
    train_generator,
    epochs=20,  # Stops early if needed
    validation_data=test_generator,
    class_weight=class_weight_dict,  # Apply class weights
    callbacks=[early_stopping, reduce_lr]
)



## STEP 8: Evaluate & Visualize Results
import matplotlib.pyplot as plt

# Extract training history values
acc = history.history['accuracy']         # Training accuracy per epoch
val_acc = history.history['val_accuracy'] # Validation accuracy per epoch
loss = history.history['loss']            # Training loss per epoch
val_loss = history.history['val_loss']    # Validation loss per epoch
epochs_range = range(len(acc))            # Number of training epochs

# Plot Accuracy (shows training/validation accuracy over epochs)
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot Loss (shows training/validation loss over epochs)
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Final Test Evaluation
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss:     {test_loss:.4f}")
print("Class Indices:", train_generator.class_indices) # Display class indicies

# Compute metrics for model analysis
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Extract true labels from test data
y_true = test_generator.classes  # true labels

# Predict on test data
y_pred_prob = model.predict(test_generator)  # Model outputs probabilities
y_pred = np.argmax(y_pred_prob, axis=1)  # Convert to class labels

# Extract class names
class_names = list(train_generator.class_indices.keys())
print("Class Names:", class_names)

# Compute confusion matrix and classification metrics
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Metrics:\n", classification_report(y_true, y_pred, target_names=class_names))
