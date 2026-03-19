#!/bin/bash

# 1. Check if an argument was provided
if [ $# -eq 0 ]; then
    echo "Error: No file path provided."
    echo "Usage: $0 path/to/file.yaml"
    exit 1
fi

FILE_PATH=$1

# 2. Check if the file exists
if [ ! -f "$FILE_PATH" ]; then
    echo "Error: File '$FILE_PATH' not found."
    exit 1
fi

# 3. Validate file extension (optional but recommended)
extension="${FILE_PATH##*.}"
if [ "$extension" != "yaml" ] && [ "$extension" != "yml" ]; then
    echo "Warning: File does not have a .yaml or .yml extension."
fi

# 4. Success logic
echo "Success: Found $FILE_PATH"
echo "Processing content..."

# 5. Ensure a virtual environment exists at venv/moscot, activate or create it
VENV_DIR="./venv/moscot"

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    echo "Found virtualenv at $VENV_DIR. Activating..."
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
else
    echo "Virtualenv not found at $VENV_DIR."
    echo "Creating virtualenv at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    if [ -f "$VENV_DIR/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
    else
        echo "Error: failed to create virtualenv at $VENV_DIR"
        exit 1
    fi

    pip install --upgrade pip
    pip install moscot
    pip install "jax[cuda12]"
    pip install -e . #makes crispy-fishstick accessible in the venv
fi

# Unset LD_LIBRARY_PATH to avoid potential conflicts with JAX
unset LD_LIBRARY_PATH

# 6. Now let's run train and test on run.py with the provided YAML file
python ./methods/moscot/run.py --yaml_config "$FILE_PATH"
