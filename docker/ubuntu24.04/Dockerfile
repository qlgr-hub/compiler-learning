# ref: Nvidia TensorRT's Dockerfile
FROM nvcr.io/nvidia/cuda:12.6.0-cudnn-devel-ubuntu24.04

# timezone settings
ENV TIME_ZONE=Asia/Shanghai 
RUN ln -snf /usr/share/zoneinfo/$TIME_ZONE /etc/localtime && echo $TIME_ZONE > /etc/timezone

# Install requried libraries
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev wget git pkg-config sudo ssh libssl-dev pbzip2 pv bzip2 unzip devscripts lintian fakeroot dh-make build-essential \
    net-tools openssh-server aria2 vim default-jre graphviz nasm libjpeg-dev libpng-dev libtiff-dev zlib1g-dev libedit-dev libxml2-dev \
    gnupg openssl clinfo pocl-opencl-icd ninja-build zip lsb-release iputils-ping device-tree-compiler libboost-regex-dev curl \
    libboost-system-dev ccache


# Delete default non-root user
RUN deluser --remove-home ubuntu

# Add default non-root sudoer user and start ssh service
RUN groupadd -r -g 1000 george && useradd -rm -d /home/george -s /bin/bash -g george -G sudo -u 1000 george
RUN echo 'george:123456' | chpasswd && echo "george ALL=(ALL) ALL" >> /etc/sudoers
RUN echo "PermitUserEnvironment yes" >> /etc/ssh/sshd_config && service ssh start


# Install python3
RUN apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-wheel &&\
    cd /usr/local/bin &&\
    ln -s /usr/bin/python3 python && \
    ln -s /usr/bin/pip3 pip


# my own proxy settings
# ENV http_proxy=http://192.168.3.126:7890
# ENV https_proxy=http://192.168.3.126:7890

USER root
WORKDIR /tmp


# Install cmake
RUN aria2c https://github.com/Kitware/CMake/releases/download/v3.31.5/cmake-3.31.5-linux-x86_64.sh && \
    bash cmake-3.31.5-linux-x86_64.sh --skip-license --prefix=/usr/local


# Install googletest
RUN git clone https://github.com/google/googletest
RUN cd googletest && mkdir build && cd build && cmake -DBUILD_SHARED_LIBS=ON .. && make && make install


# Install OpenCL
RUN git clone --recursive https://github.com/KhronosGroup/OpenCL-SDK.git
WORKDIR /tmp/OpenCL-SDK
RUN cmake -DBUILD_TESTING=OFF \
      -DBUILD_DOCS=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DOPENCL_SDK_BUILD_SAMPLES=OFF \
      -DOPENCL_SDK_TEST_SAMPLES=OFF \
      -B ./build -S .
RUN cmake --build ./build --target install
RUN cp -r ./install/include/* /usr/local/include
RUN cp ./install/lib/libOpenCL.so /usr/local/lib/


# Get miniconda
USER george
WORKDIR /home/george
RUN aria2c https://repo.anaconda.com/miniconda/Miniconda3-py312_25.1.1-0-Linux-x86_64.sh
RUN bash Miniconda3-py312_25.1.1-0-Linux-x86_64.sh -b -u


# conda init and conda update
RUN /home/george/miniconda3/bin/conda init
RUN /home/george/miniconda3/bin/conda config --add channels conda-forge
RUN /home/george/miniconda3/bin/conda config --set auto_activate_base false
RUN /home/george/miniconda3/bin/conda update --all
RUN /home/george/miniconda3/bin/conda clean -ya
RUN /home/george/miniconda3/bin/conda install conda-build conda-verify

WORKDIR /home/george

# use local build-environment.yml instead
ADD build-environment.yaml build-environment.yaml
RUN /home/george/miniconda3/bin/conda env create --file build-environment.yaml
RUN /home/george/miniconda3/bin/conda clean -ya

ENV TVM_HOME=/workspace/opensource/tvm/
ENV TVM_CONFIGS_JSON_DIR=${TVM_HOME}/configs/host/
ENV TVM_LOG_DEBUG="ir/transform.cc=1,relay/ir/transform.cc=1"
ENV PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
ENV PATH=${PATH}:/usr/local/cuda/bin/
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64/


# Clean up
USER root
RUN rm -rvf /tmp/* && rm /home/george/Miniconda3-py312_25.1.1-0-Linux-x86_64.sh


EXPOSE 22
EXPOSE 8888
WORKDIR /home/george
ADD startup.sh startup.sh
ENTRYPOINT ["bash", "startup.sh"]
