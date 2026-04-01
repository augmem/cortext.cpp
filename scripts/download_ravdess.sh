#!/usr/bin/env bash
# Download RAVDESS Audio_Speech_Actors_01-24.zip
# Usage: ./scripts/download_ravdess.sh [output_dir]
#
# RAVDESS: Ryerson Audio-Visual Database of Emotional Speech and Song
# Source: https://zenodo.org/record/1188976

set -euo pipefail

OUTPUT_DIR="${1:-$(dirname "$0")/../data/ravdess}"
RAVDESS_URL="https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
ZIP_FILE="$OUTPUT_DIR/Audio_Speech_Actors_01-24.zip"

mkdir -p "$OUTPUT_DIR"

if [ -d "$OUTPUT_DIR/Audio_Speech_Actors_01-24" ]; then
    echo "RAVDESS already extracted at $OUTPUT_DIR/Audio_Speech_Actors_01-24"
    exit 0
fi

if [ -f "$ZIP_FILE" ]; then
    echo "Using existing zip file: $ZIP_FILE"
else
    echo "Downloading RAVDESS to $OUTPUT_DIR..."
    curl -L "$RAVDESS_URL" -o "$ZIP_FILE"
fi

echo "Extracting..."
unzip -q "$ZIP_FILE" -d "$OUTPUT_DIR"

echo "Done. Files in $OUTPUT_DIR/Audio_Speech_Actors_01-24/"
echo "Total audio files: $(find "$OUTPUT_DIR/Audio_Speech_Actors_01-24" -name "*.wav" | wc -l | tr -d ' ')"
