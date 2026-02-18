#!/bin/bash
echo "Building Docker image..."
docker build -t turbofan-api:latest .
echo "Build complete."
