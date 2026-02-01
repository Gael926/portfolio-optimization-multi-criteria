#!/bin/bash

PORT=8501
MAX_PORT=8510

echo "--- Finance App Docker Runner ---"

while [ $PORT -le $MAX_PORT ]; do
    if ! lsof -i :$PORT > /dev/null; then
        break
    fi
    echo "Port $PORT is already in use, trying next one..."
    PORT=$((PORT+1))
done

if [ $PORT -gt $MAX_PORT ]; then
    echo "Error: No free ports found between 8501 and $MAX_PORT."
    exit 1
fi

echo "Starting application on http://localhost:$PORT"
export HOST_PORT=$PORT
docker compose up --build
