#!/bin/bash

# Download and extract data from Dropbox
# Usage: ./download_and_extract.sh [output_directory]

OUTPUT_DIR="${1:-.}"  # Default to current directory if not specified

# Dropbox URL (modified for direct download)
URL="https://www.dropbox.com/scl/fi/gq7jr8dbsrmyflpbdbnj0/data.tar.gz?rlkey=0bdjk230ef5nylg26gg9tl4nh&st=ugsmk6wt&dl=1"

FILENAME="data.tar.gz"

echo "Downloading $FILENAME..."
wget -O "$FILENAME" "$URL" --no-check-certificate

if [ $? -eq 0 ]; then
    echo "Download complete. Extracting..."
    tar -xzf "$FILENAME" -C "$OUTPUT_DIR"

    if [ $? -eq 0 ]; then
        echo "Extraction complete!"
        echo "Contents extracted to: $OUTPUT_DIR"

        # Optionally remove the archive after extraction
        # rm "$FILENAME"
        # echo "Archive removed."
    else
        echo "Error: Extraction failed."
        exit 1
    fi
else
    echo "Error: Download failed."
    exit 1
fi