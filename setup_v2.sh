#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define color variables
YELLOW='\033[1;33m'
NC='\033[0m' # No Color (resets the color back to normal)

# Load environment variables from .env file
echo -e "${YELLOW}Loading environment variables from .env file...${NC}"
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
  export HF_TOKEN=$HUGGINGFACE_TOKEN
  echo "Hugging Face token loaded from .env"
else
  echo "Error: .env file not found"
  exit 1
fi

# Configure Git settings
echo -e "${YELLOW}Configuring Git...${NC}"
git config --global credential.helper store
git config --global user.email "bdragun@villanova.edu"
git config --global user.name "Brian Dragun"

echo -e "${YELLOW}Updating system packages...${NC}"
sudo apt-get update && sudo apt-get upgrade -y

echo -e "${YELLOW}Installing dependencies...${NC}"
sudo apt-get install -y pybind11-dev libopenmpi-dev

# Set CPLUS_INCLUDE_PATH for pybind11 (optional)
echo -e "${YELLOW}Setting CPLUS_INCLUDE_PATH for pybind11...${NC}"
export CPLUS_INCLUDE_PATH=$(python -m pybind11 --includes | sed 's/-I//g' | tr ' ' ':')

# Install Python dependencies
echo -e "${YELLOW}Installing Python requirements...${NC}"
pip install -r requirements.txt
pip uninstall -y numpy
pip install 'numpy<2.0'

# Ensure the Hugging Face Hub library is installed
echo -e "${YELLOW}Ensuring huggingface_hub is installed...${NC}"
pip install huggingface_hub --upgrade

# Hugging Face Login using Python API
echo -e "${YELLOW}Logging into Hugging Face using Python API...${NC}"
python3 - <<EOF
import os
from huggingface_hub import login

token = os.getenv("HF_TOKEN")  # Retrieve token from environment
if token:
    login(token)
    print("Hugging Face login successful!")
else:
    print("Error: HF_TOKEN is not set.")
EOF

# Install Oh My Posh
echo -e "${YELLOW}Installing Oh My Posh...${NC}"
curl -s https://ohmyposh.dev/install.sh | bash -s

# Install Nerd Font for Oh My Posh
echo -e "${YELLOW}Installing Nerd Font (Hack)...${NC}"
wget https://github.com/ryanoasis/nerd-fonts/releases/download/v3.1.1/Hack.zip
mkdir -p ~/.local/share/fonts
unzip Hack.zip -d ~/.local/share/fonts
fc-cache -fv

# Clean up downloaded zip file
echo -e "${YELLOW}Cleaning up downloaded font archive...${NC}"
rm Hack.zip

# Download and install Oh My Posh themes
echo -e "${YELLOW}Installing Oh My Posh themes...${NC}"
mkdir -p ~/.poshthemes
wget https://github.com/JanDeDobbeleer/oh-my-posh/releases/latest/download/themes.zip -O ~/.poshthemes/themes.zip
unzip ~/.poshthemes/themes.zip -d ~/.poshthemes
chmod u+rw ~/.poshthemes/*.json
rm ~/.poshthemes/themes.zip

# Configure Oh My Posh for the current shell
echo -e "${YELLOW}Configuring Oh My Posh...${NC}"
if [ -f ~/.bashrc ]; then
  # Check if Oh My Posh is already in .bashrc to avoid duplicates
  if grep -q "oh-my-posh init bash" ~/.bashrc; then
    # Replace existing Oh My Posh configuration
    sed -i 's|eval "$(oh-my-posh init bash.*)".*|eval "$(oh-my-posh init bash --config ~/.poshthemes/powerline.omp.json)"|g' ~/.bashrc
    echo "Oh My Posh theme updated in ~/.bashrc"
  else
    # Add new Oh My Posh configuration
    echo 'eval "$(oh-my-posh init bash --config ~/.poshthemes/powerline.omp.json)"' >> ~/.bashrc
    echo "Oh My Posh added to ~/.bashrc"
  fi
fi

# If ZSH is installed, also configure for it
if [ -f ~/.zshrc ]; then
  if grep -q "oh-my-posh init zsh" ~/.zshrc; then
    # Replace existing Oh My Posh configuration
    sed -i 's|eval "$(oh-my-posh init zsh.*)".*|eval "$(oh-my-posh init zsh --config ~/.poshthemes/powerline.omp.json)"|g' ~/.zshrc
    echo "Oh My Posh theme updated in ~/.zshrc"
  else
    # Add new Oh My Posh configuration
    echo 'eval "$(oh-my-posh init zsh --config ~/.poshthemes/powerline.omp.json)"' >> ~/.zshrc
    echo "Oh My Posh added to ~/.zshrc"
  fi
fi

echo -e "${YELLOW}Setup complete. You may need to restart your terminal or run 'source ~/.bashrc' to apply Oh My Posh changes.${NC}"