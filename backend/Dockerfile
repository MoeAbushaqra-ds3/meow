FROM python:3.9

ENV OPENCV_VERSION=4.4.0 \
    OPENCV_PREFIX=/opt/opencv

WORKDIR /app

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
        postgresql-client \
        libglib2.0-0 \
        libgl1 \
        ffmpeg \
        build-essential \
        cmake \
        pkg-config \
        curl \
        ca-certificates \
        git \
        libjpeg62-turbo-dev \
        libpng-dev \
        libtiff-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        libopenexr-dev \
        liblapacke-dev \
        gfortran \
        libgtk-3-dev \
        libcanberra-gtk3-module \
        libdc1394-dev \
        ninja-build && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz -o /tmp/opencv.tar.gz && \
    curl -sSL https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.tar.gz -o /tmp/opencv_contrib.tar.gz && \
    tar -xzf /tmp/opencv.tar.gz -C /tmp && \
    tar -xzf /tmp/opencv_contrib.tar.gz -C /tmp && \
    cmake -S /tmp/opencv-${OPENCV_VERSION} -B /tmp/opencv-build \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_INSTALL_PREFIX=${OPENCV_PREFIX} \
        -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib-${OPENCV_VERSION}/modules \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_DOCS=OFF \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_opencv_python2=OFF \
        -D BUILD_opencv_python3=OFF \
        -D BUILD_opencv_java=OFF && \
    cmake --build /tmp/opencv-build --target install -j"$(nproc)" && \
    echo "${OPENCV_PREFIX}/lib" > /etc/ld.so.conf.d/opencv.conf && \
    ldconfig && \
    rm -rf /tmp/opencv.tar.gz /tmp/opencv_contrib.tar.gz /tmp/opencv-${OPENCV_VERSION} /tmp/opencv_contrib-${OPENCV_VERSION} /tmp/opencv-build

COPY requirements.txt ./project-requirements.txt
COPY backend/requirements.txt ./backend-requirements.txt
RUN pip install --no-cache-dir -r project-requirements.txt -r backend-requirements.txt

COPY backend/ .
COPY ml ./ml
COPY stitching ./stitching

RUN cmake -S stitching -B stitching/build-linux \
        -DCMAKE_BUILD_TYPE=Release \
        -DOpenCV_DIR=${OPENCV_PREFIX}/lib/cmake/opencv4 && \
    cmake --build stitching/build-linux --target image-stitching --config Release && \
    install -m 755 stitching/build-linux/image-stitching /usr/local/bin/image-stitching && \
    rm -rf stitching/build-linux

EXPOSE 8888

CMD ["python", "-m", "app.main"]
