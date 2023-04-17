#!/bin/bash

# Set the download URL base
URL_BASE="https://storage.googleapis.com/ai2-jackh-mmc4-public/data_core/docs_no_face_shard_"

# Set the folder where you want to save the unzipped files
DESTINATION_FOLDER="/path/to/destination/folder"

# Create the destination folder if it doesn't exist
mkdir -p "$DESTINATION_FOLDER"

# Loop through the shard numbers and download and unzip the files
for SHARD in {0..23098}; do
  URL="${URL_BASE}${SHARD}_v3.jsonl.zip"
  echo "Downloading shard $SHARD from $URL..."

  # Download the file (continue if the file is missing or there is an error)
  curl -fsSL --retry 3 --retry-delay 5 --max-time 20 --continue-at - "$URL" -o "shard_${SHARD}.zip" || echo "Error downloading shard $SHARD, continuing..."

  # Unzip the file if it was downloaded successfully
  if [ -f "shard_${SHARD}.zip" ]; then
    echo "Unzipping shard_${SHARD}.zip to $DESTINATION_FOLDER..."
    unzip -q "shard_${SHARD}.zip" -d "$DESTINATION_FOLDER"

    # Remove the zip file after unzipping
    rm "shard_${SHARD}.zip"
  fi
done

echo "Download and unzip process completed."