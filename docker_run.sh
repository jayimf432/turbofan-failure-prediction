#!/bin/bash
echo "Starting services with Docker Compose..."
docker-compose up -d --build
echo "Services started. API available at http://localhost:8000"
echo "Use 'docker-compose logs -f' to view logs."
