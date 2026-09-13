FROM python:3.11-slim

# Install system dependencies for C++ compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the C++ source and Python build files
COPY CMakeLists.txt setup.py pyproject.toml ./
COPY include/ ./include/
COPY src/ ./src/

# Install the vectorforge package (this triggers the CMake C++ build)
RUN pip install .

# Copy the server code
COPY server/ ./server/

# Install server dependencies
RUN pip install -r server/requirements.txt

# Expose the API port
EXPOSE 8000

# Set the entrypoint to the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
