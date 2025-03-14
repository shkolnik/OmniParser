FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Install Python 3 and pip
RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip \
        python-is-python3 \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the application code to the container
COPY ./requirements.txt /app

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

RUN mkdir /app/weights && \
    for folder in icon_caption icon_detect; \
        do huggingface-cli download microsoft/OmniParser-v2.0 --local-dir weights --repo-type model --include "$folder/*"; \
    done && \
    mv weights/icon_caption weights/icon_caption_florence

COPY ./util /app/util
COPY ./omnitool/omniparserserver/omniparserserver.py /app/omnitool/omniparserserver/

WORKDIR /app/omnitool/omniparserserver

# Start server once to download and cache model files
RUN OMNIPARSER_DOWNLOAD_ONLY=1 python omniparserserver.py

# Command to run the application
EXPOSE 8000
CMD ["python", "-m", "omniparserserver"]