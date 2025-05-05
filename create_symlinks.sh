#!/bin/bash

# Define color variables
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Creating symbolic links for CUDA libraries...${NC}"

# Directory where bitsandbytes libraries are stored
BNB_DIR=$(pip show bitsandbytes | grep Location | awk '{print $2}')/bitsandbytes

if [ -d "$BNB_DIR" ]; then
    echo -e "${GREEN}Found bitsandbytes directory: ${BNB_DIR}${NC}"

    # Create symlinks for CUDA 12.8 if not exists
    if [ ! -f "${BNB_DIR}/libbitsandbytes_cuda128.so" ]; then
        echo -e "${YELLOW}Creating symlink for CUDA 12.8 using CUDA 12.3 library...${NC}"
        if [ -f "${BNB_DIR}/libbitsandbytes_cuda123.so" ]; then
            ln -sf "${BNB_DIR}/libbitsandbytes_cuda123.so" "${BNB_DIR}/libbitsandbytes_cuda128.so"
            echo -e "${GREEN}Created symlink: ${BNB_DIR}/libbitsandbytes_cuda128.so -> ${BNB_DIR}/libbitsandbytes_cuda123.so${NC}"
        elif [ -f "${BNB_DIR}/libbitsandbytes_cuda122.so" ]; then
            ln -sf "${BNB_DIR}/libbitsandbytes_cuda122.so" "${BNB_DIR}/libbitsandbytes_cuda128.so"
            echo -e "${GREEN}Created symlink: ${BNB_DIR}/libbitsandbytes_cuda128.so -> ${BNB_DIR}/libbitsandbytes_cuda122.so${NC}"
        else
            echo -e "${YELLOW}No compatible CUDA library found. Will try to use CPU fallback.${NC}"
        fi
    fi

    # Create symbolic link for libcudart.so if needed
    echo -e "${YELLOW}Setting up libcudart.so symbolic links...${NC}"
    if [ -f "/usr/lib/aarch64-linux-gnu/libcudart.so.12" ] && [ ! -f "/usr/lib/aarch64-linux-gnu/libcudart.so" ]; then
        sudo ln -sf /usr/lib/aarch64-linux-gnu/libcudart.so.12 /usr/lib/aarch64-linux-gnu/libcudart.so
        echo -e "${GREEN}Created symlink: /usr/lib/aarch64-linux-gnu/libcudart.so -> /usr/lib/aarch64-linux-gnu/libcudart.so.12${NC}"
    fi

    # Tell bitsandbytes to use the latest CUDA version
    echo -e "${YELLOW}Setting BNB_CUDA_VERSION environment variable...${NC}"
    export BNB_CUDA_VERSION=128
    echo "export BNB_CUDA_VERSION=128" >> ~/.bashrc
else
    echo -e "${YELLOW}Could not find bitsandbytes directory. Make sure it's installed correctly.${NC}"
fi

echo -e "\n${GREEN}Done creating symlinks. Now apply the changes:${NC}"
echo "source ~/.bashrc"