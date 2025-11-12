FROM python:3.9

WORKDIR /app

RUN apt update && \
    apt install -y build-essential cmake pkg-config libopencv-dev postgresql-client libglib2.0-0 libgl1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./project-requirements.txt
COPY backend/requirements.txt ./backend-requirements.txt
RUN pip install --no-cache-dir -r project-requirements.txt -r backend-requirements.txt

COPY backend/ .
COPY ml/ ./ml
COPY stitching/ ./stitching

# Build the native stitching binary so the API can invoke it at runtime
RUN rm -rf /app/stitching/build && \
    cmake -S /app/stitching -B /app/stitching/build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /app/stitching/build --target image-stitching --config Release

# Ensure any prebuilt stitching binary is executable when present
RUN if [ -f /app/stitching/build/image-stitching ]; then \
        chmod +x /app/stitching/build/image-stitching; \
    fi

EXPOSE 8888

CMD ["python", "-m", "app.main"]
