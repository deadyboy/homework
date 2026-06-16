#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-facades}"
case "$DATASET" in
  cityscapes|night2day|edges2handbags|edges2shoes|facades|maps) ;;
  *)
    echo "Available datasets: cityscapes, night2day, edges2handbags, edges2shoes, facades, maps" >&2
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/datasets"
TAR_FILE="$DATA_DIR/$DATASET.tar.gz"
TARGET_DIR="$DATA_DIR/$DATASET"
URL="http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/$DATASET.tar.gz"

mkdir -p "$DATA_DIR"
echo "Downloading $DATASET from $URL"
wget -N "$URL" -O "$TAR_FILE"
tar -zxf "$TAR_FILE" -C "$DATA_DIR"
rm "$TAR_FILE"

find "$TARGET_DIR/train" -type f \( -name "*.jpg" -o -name "*.png" \) | sort -V | sed "s#^$SCRIPT_DIR/##" > "$SCRIPT_DIR/train_list.txt"
find "$TARGET_DIR/val" -type f \( -name "*.jpg" -o -name "*.png" \) | sort -V | sed "s#^$SCRIPT_DIR/##" > "$SCRIPT_DIR/val_list.txt"

echo "Wrote train_list.txt and val_list.txt for $DATASET"
