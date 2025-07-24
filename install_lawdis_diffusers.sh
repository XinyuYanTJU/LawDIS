#!/bin/bash

# Step 0: Locate the installed diffusers package path
DIFFUSERS_SITE_PACKAGES=$(pip show diffusers | grep "Location" | awk '{print $2}')
DIFFUSERS_PATH="${DIFFUSERS_SITE_PACKAGES}/diffusers"

echo "Detected diffusers installation path: $DIFFUSERS_PATH"

# Step 1: Copy custom VAE implementation files
cp diffusers_lawdis/models/autoencoders/autoencoder_kl_lawdis.py "${DIFFUSERS_PATH}/models/autoencoders/"
cp diffusers_lawdis/models/autoencoders/vae_lawdis.py "${DIFFUSERS_PATH}/models/autoencoders/"
echo "Copied autoencoder_kl_lawdis.py and vae_lawdis.py to diffusers/models/autoencoders/"

# Step 2: Replace autoencoders/__init__.py
cp diffusers_lawdis/models/autoencoders/__init__.py "${DIFFUSERS_PATH}/models/autoencoders/__init__.py"
echo "Replaced diffusers/models/autoencoders/__init__.py"

# Step 3: Replace models/__init__.py
cp diffusers_lawdis/models/__init__.py "${DIFFUSERS_PATH}/models/__init__.py"
echo "Replaced diffusers/models/__init__.py"

# Step 4: Replace diffusers/__init__.py
cp diffusers_lawdis/__init__.py "${DIFFUSERS_PATH}/__init__.py"
echo "Replaced diffusers/__init__.py"

echo "LawDIS custom VAE integration complete!"
