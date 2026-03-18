#!/bin/bash
echo "Starting Paper2Any Backend..."
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
