#!/bin/bash
# Download only SleepEEG and Epilepsy datasets

echo "📥 Downloading SleepEEG and Epilepsy datasets..."

# Download SleepEEG if not already present
if [ ! -f "SleepEEG.zip" ]; then
    echo "Downloading SleepEEG dataset..."
    wget --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" -O SleepEEG.zip https://figshare.com/ndownloader/articles/19930178/versions/1
else
    echo "SleepEEG.zip already exists, skipping download"
fi

# Download Epilepsy if not already present
if [ ! -f "Epilepsy.zip" ]; then
    echo "Downloading Epilepsy dataset..."
    wget --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" -O Epilepsy.zip https://figshare.com/ndownloader/articles/19930199/versions/2
else
    echo "Epilepsy.zip already exists, skipping download"
fi

# Unzip datasets
echo ""
echo "📦 Extracting datasets..."
unzip -o SleepEEG.zip -d src/tfc/data/SleepEEG/
unzip -o Epilepsy.zip -d src/tfc/data/Epilepsy/

# Clean up zip files
echo ""
echo "🧹 Cleaning up..."
rm SleepEEG.zip Epilepsy.zip

echo "✅ Done! SleepEEG and Epilepsy datasets are ready."