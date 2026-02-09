# Use an official Python runtime as a parent image
FROM python:3.10-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Suppress the "Running pip as the 'root' user" warning
ENV PIP_ROOT_USER_ACTION=ignore

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Render uses the PORT environment variable to define the port the service should listen on.
# We will use this for the Streamlit frontend. 
# FastAPI will run internally on port 8000.
EXPOSE 8000

# Create a startup script to run both servers
# We bind Streamlit to the PORT env var and FastAPI to localhost
RUN echo '#!/bin/bash\n\
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000 & \n\
streamlit run app/app.py --server.port ${PORT:-8501} --server.address 0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

# Run the startup script
CMD ["/app/start.sh"]
