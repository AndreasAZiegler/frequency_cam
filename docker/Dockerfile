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

# Install ROS dependencies
RUN apt-get install -y ros-galactic-camera-info-manager*

# Install calibration dependencies
RUN apt-get install -y libceres-dev libeigen3-dev gdb
