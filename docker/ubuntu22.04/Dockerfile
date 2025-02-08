# ref:https://github.com/NVIDIA/TensorRT/blob/release/10.0/docker/ubuntu-22.04.Dockerfile
ARG CUDA_VERSION=12.2.2
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04
# Required to build Ubuntu 20.04 without user prompts with DLFW container
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai


# Install cudnn
ENV NV_CUDNN_VERSION 8.9.6.50
ENV NV_CUDNN_PACKAGE_NAME "libcudnn8"
ENV CUDA_VERSION_MAJOR_MINOR=12.2
ENV NV_CUDNN_PACKAGE "libcudnn8=$NV_CUDNN_VERSION-1+cuda${CUDA_VERSION_MAJOR_MINOR}"
ENV NV_CUDNN_PACKAGE_DEV "libcudnn8-dev=$NV_CUDNN_VERSION-1+cuda${CUDA_VERSION_MAJOR_MINOR}"
ENV TRT_VERSION 10.0.1.6
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    ${NV_CUDNN_PACKAGE} \
    ${NV_CUDNN_PACKAGE_DEV} \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} \
    && rm -rf /var/lib/apt/lists/*


# Update CUDA signing key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub


# Install requried libraries
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev wget git pkg-config sudo ssh libssl-dev pbzip2 pv bzip2 unzip devscripts lintian fakeroot dh-make build-essential \
    net-tools openssh-server aria2 vim default-jre graphviz nasm libjpeg-dev libpng-dev libtiff-dev zlib1g-dev libedit-dev libxml2-dev \
    gnupg openssl clinfo pocl-opencl-icd ninja-build zip lsb-release iputils-ping


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
ENV http_proxy=http://192.168.3.126:7890
ENV https_proxy=http://192.168.3.126:7890

USER root
WORKDIR /tmp


# Install llvm
WORKDIR /tmp
RUN aria2c https://apt.llvm.org/llvm.sh  && chmod +x llvm.sh && ./llvm.sh 18
ENV CC=clang-18
ENV CXX=clang++-18


# Install cmake
RUN aria2c https://github.com/Kitware/CMake/releases/download/v3.30.0/cmake-3.30.0-linux-x86_64.sh && \
    bash cmake-3.30.0-linux-x86_64.sh --skip-license --prefix=/usr/local


# Install googletest
RUN git clone https://github.com/google/googletest
RUN cd googletest && mkdir build && cd build && cmake -DBUILD_SHARED_LIBS=ON .. && make && make install


# Install OpenCL
RUN git clone --recursive --depth 1 --branch v2022.09.30 https://github.com/KhronosGroup/OpenCL-SDK.git
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
RUN aria2c https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh
RUN bash Miniconda3-py310_24.7.1-0-Linux-x86_64.sh -b -u


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
RUN rm -rvf /tmp/* && rm /home/george/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh


EXPOSE 22
EXPOSE 8888
WORKDIR /home/george
ADD startup.sh startup.sh
ENTRYPOINT ["bash", "startup.sh"]
