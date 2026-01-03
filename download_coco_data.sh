#!/bin/bash

# 1. Create the data directory
echo "Creating 'data' directory..."
mkdir -p data
cd data

# 2. Download Validation Images (val2017)
# Check if wget is installed, otherwise use curl
echo "Downloading val2017 images (approx 1GB)..."
if command -v wget &> /dev/null; then
    wget -c http://images.cocodataset.org/zips/val2017.zip
else
    curl -L -O http://images.cocodataset.org/zips/val2017.zip
fi

# 3. Download Annotations
echo "Downloading annotations (approx 241MB)..."
if command -v wget &> /dev/null; then
    wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
else
    curl -L -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
fi

# 4. Unzip files
echo "Unzipping val2017..."
unzip -q val2017.zip

echo "Unzipping annotations..."
unzip -q annotations_trainval2017.zip

# 5. Cleanup zip files
echo "Cleaning up zip files..."
rm val2017.zip annotations_trainval2017.zip

echo "------------------------------------------------"
echo "Download complete."
echo "Images are in: $(pwd)/val2017"
echo "Annotations are in: $(pwd)/annotations"