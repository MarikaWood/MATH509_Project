##############################################
# STEP 1: Install & Configure Kaggle CLI
##############################################
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

##############################################
# STEP 2: Download & Unzip the Dataset
##############################################
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
