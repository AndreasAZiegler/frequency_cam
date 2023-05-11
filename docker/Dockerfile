FROM osrf/ros:galactic-desktop

ENV DEBIAN_FRONTEND noninteractive

# Some system dependencies
RUN apt-get update
RUN apt-get install -y python3-wstool ros-galactic-ament-cmake-clang-format gdb less

# Spinnaker-sdk
# Dependency for the Spinnaker-sdk
RUN apt-get install -y qt5-default
## IMPORTANT: Copy spinnaker-2.5.0.80-amd64-Ubuntu20.04 from /cshome/share/TT_robot/debs/ to the docker folder and rename it to spinnaker-2.5.0.80-amd64
ADD spinnaker-3.0.0.118-amd64 /root/spinnaker-3.0.0.118-amd64
RUN cd /root/spinnaker-3.0.0.118-amd64 && \
    echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && \
    echo libspinnaker libspinnaker/accepted-flir-eula boolean true | debconf-set-selections && \
    printf "Y\nn\nn\nn\nn\nn\n" | ./install_spinnaker.sh && \
    cd .. && rm -R spinnaker-3.0.0.118-amd64

# OpenCV
RUN apt-get install -y build-essential cmake pkg-config unzip yasm git checkinstall wget
RUN apt-get install -y libjpeg-dev libpng-dev libtiff-dev
RUN apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libavresample-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev libfaac-dev libmp3lame-dev libvorbis-dev
RUN apt-get install -y libdc1394-22 libdc1394-22-dev libxine2-dev libv4l-dev v4l-utils
RUN cd /usr/include/linux && ln -s -f ../libv4l1-videodev.h videodev.h
RUN apt-get install -y libgtk-3-dev
RUN apt-get install -y libtbb-dev
RUN apt-get install -y libatlas-base-dev gfortran
RUN apt-get install -y libprotobuf-dev protobuf-compiler libgoogle-glog-dev libgflags-dev libgphoto2-dev libeigen3-dev libhdf5-dev doxygen

RUN cd ~ && \
    wget -O opencv.zip https://github.com/opencv/opencv/archive/4.7.0.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.7.0.zip && \
    unzip opencv.zip && \
    unzip opencv_contrib.zip && \
    cd opencv-4.7.0 && \
    mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_C_COMPILER=/usr/bin/gcc-9 \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D INSTALL_PYTHON_EXAMPLES=ON \
          -D INSTALL_C_EXAMPLES=OFF \
          -D WITH_TBB=ON \
          -D BUILD_opencv_cudacodec=OFF \
          -D ENABLE_FAST_MATH=1 \
          -D CUDA_FAST_MATH=1 \
          -D WITH_CUDA=OFF \
          -D WITH_CUBLAS=OFF \
          -D WITH_V4L=ON \
          -D WITH_QT=OFF \
          -D WITH_OPENGL=ON \
          -D WITH_GSTREAMER=ON \
          -D OPENCV_GENERATE_PKGCONFIG=ON \
          -D OPENCV_PC_FILE_NAME=opencv.pc \
          -D OPENCV_ENABLE_NONFREE=ON \
          -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-4.7.0/modules \
          -D PYTHON_EXECUTABLE=~/.virtualenvs/cv/bin/python \
          -D BUILD_TESTS=ON \
          -D BUILD_PERF_TESTS=ON \
          -D BUILD_EXAMPLES=OFF \
          -D WITH_CUDNN=OFF \
          -D OPENCV_DNN_CUDA=OFF \
          -D CUDA_ARCH_BIN=7.5 .. && \
    make -j 6 && \
    make install

# Install ROS dependencies
RUN apt-get install -y ros-galactic-camera-info-manager*

# Install calibration dependencies
RUN apt-get install -y libceres-dev libeigen3-dev gdb
