##########################################
# STEP 1: Install & Configure Kaggle CLI
##########################################
!pip install -q kaggle

import os

# Tell Kaggle CLI to look for the API key in /content
os.environ["KAGGLE_CONFIG_DIR"] = "/content"

# Create a ~/.kaggle folder
!mkdir -p ~/.kaggle

# Manually upload kaggle.json file
uploaded = files.upload()  # select kaggle.json when prompted

# Move kaggle.json into ~/.kaggle folder
!cp /content/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

########################################
# STEP 2: Download & Unzip the Dataset
########################################
# Downloads the Images Oasis dataset from Kaggle
!kaggle datasets download ninadaithal/imagesoasis -p /content --force

# Unzip into /content/Data
!unzip -q /content/imagesoasis.zip -d /content

# Expected structure:
# /content/Data/
#   Mild Dementia/
#   Moderate Dementia/
#   Non Demented/
#   Very Mild Dementia/

# Each folder has a set of slices for various patients, e.g.:
#   OAS1_0285_MR1_mpr-1_128.jpg (0285 = patient, MR1 = 1st MRI, 128 = MRI slice)

######################################################
# STEP 3: Filter to Use Only 150th Slice Per Patient
######################################################
import shutil
import re

# Define root directory where dataset is stored
data_root = "/content/Data"

# Define the directory for filtered images
filtered_root = "/content/Data_150th_Slice"
os.makedirs(filtered_root, exist_ok=True)

# Classes (subfolders)
classes = ["Mild Dementia", "Moderate Dementia", "Non Demented", "Very mild Dementia"]

# Regex to extract the slice number from filenames
slice_pattern = re.compile(r".*_(\d+)\.jpg$")

for cls in classes:
    class_in_path = os.path.join(data_root, cls)
    class_out_path = os.path.join(filtered_root, cls)
    os.makedirs(class_out_path, exist_ok=True)

    for fname in os.listdir(class_in_path):
        match = slice_pattern.match(fname)
        if match:
            slice_idx = int(match.group(1))
            # Only keep the 150th slice of each patient
            if slice_idx == 150:
                src = os.path.join(class_in_path, fname)
                dst = os.path.join(class_out_path, fname)
                shutil.copy2(src, dst)

print("Data filtered to keep only 150th slice for each patient.")

#################################################
# STEP 4: Split the Data into Train/Test (70:30)
#################################################
import random

base_dir = "/content/OASIS_split" # Root directory for split dataset
train_dir = os.path.join(base_dir, "train") # Path for training data
test_dir  = os.path.join(base_dir, "test") # Path for testing data

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True) # Makes sure train/test directories exist

# For each class, split the data into training (70%) and testing (30%) sets
for cls in classes:
    cls_folder = os.path.join(filtered_root, cls)
    all_images = os.listdir(cls_folder)
    random.shuffle(all_images)

    # 70:30 split
    split_idx = int(0.7 * len(all_images))
    train_imgs = all_images[:split_idx] # First 70% training
    test_imgs  = all_images[split_idx:] # Last 30% testing

    # Make subfolders for each class in train/test directories
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    # Copy the training images
    for img_name in train_imgs:
        src = os.path.join(cls_folder, img_name)
        dst = os.path.join(train_dir, cls, img_name)
        shutil.copy2(src, dst)

    # Copy the testing images
    for img_name in test_imgs:
        src = os.path.join(cls_folder, img_name)
        dst = os.path.join(test_dir, cls, img_name)
        shutil.copy2(src, dst)

print("Data split complete.")
print("Train folder:", train_dir)
print("Test folder:", test_dir)

#####################################
# STEP 5: Keras ImageDataGenerators
#####################################
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (64, 64) # Reduced image size for faster training
BATCH_SIZE = 32 # Number of images processed per training step

# Training data generator with augmentations
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Testing data generator (just rescale, no augmentation)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir, # Load images from training directory
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical' # Multiclass classification
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # Keep image ordering consistent
)

##############################################
# STEP 6: Use MobileNetV2 (Transfer Learning)
##############################################
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# Load the MobileNetV2 model, pre-trained on ImageNet, excluding the top layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
base_model.trainable = False  # Freeze base model layers

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3), # Dropout to prevent overfitting
    layers.Dense(4, activation='softmax')  # 4 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

##############################################
# STEP 7: Train the Model with Early Stopping
##############################################
# Early stopping to stop training if validation loss doesn't improve
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with small number of epochs to start 
history = model.fit(
    train_generator,
    epochs=3,  # 3 epochs selected
    validation_data=test_generator,
    callbacks=[early_stopping]
)

########################################
# STEP 8: Evaluate & Visualize Results
########################################
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(3)  # 3 epochs

# Plot Accuracy
plt.figure()
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()
plt.show()

# Plot Loss
plt.figure()
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.show()

# Final Test Evaluation
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss:     {test_loss:.4f}")

print("Class Indices:", train_generator.class_indices)
