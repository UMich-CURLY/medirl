FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-devel
ENV TZ=US/Michigan
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install -y \
    git \
    x11-apps \
    mesa-utils \
    libgl1-mesa-glx \
    apt-utils \
    python \
    wget \
    lsb-release \
    git 

RUN apt-get update && apt-get -y install rsync

RUN pip install cython numpy==1.15.4 
RUN pip install numba==0.50.1 matplotlib==2.1.2 
RUN pip install scipy==1.0.0 seaborn==0.9.0 
RUN pip install tqdm==4.19.4 visdom==0.1.8.3
RUN pip install matplotlib
RUN export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64"

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# RUN curl -L -k -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
#     chmod +x ~/miniconda.sh &&\
#     ~/miniconda.sh -b -p /opt/conda &&\
#     rm ~/miniconda.sh &&\
#     /opt/conda/bin/conda install numpy pyyaml scipy ipython mkl mkl-include &&\
#     /opt/conda/bin/conda clean -ya
# ENV PATH /opt/conda/bin:$PATH

# # Install cmake
# RUN wget https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0-Linux-x86_64.sh
# RUN mkdir /opt/cmake
# RUN sh /cmake-3.14.0-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
# RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
# RUN cmake --version


# RUN apt-get update && apt-get install -y lsb-release && apt-get clean all
RUN conda config --set ssl_verify false
RUN conda create -n robostackenv python=3.9 -c conda-forge
RUN /bin/bash -c ". activate robostackenv; conda config --env --add channels conda-forge; conda config --env --add channels robostack-experimental; conda config --env --add channels robostack"
RUN /bin/bash -c ". activate robostackenv; conda install -y ros-noetic-desktop"
# RUN /bin/bash -c ". activate robostackenv; conda install -y ros-noetic-map-server"
# RUN /bin/bash -c ". activate robostackenv; conda install -y ros-noetic-move-base"
# RUN /bin/bash -c ". activate robostackenv; conda install -y ros-noetic-amcl"

WORKDIR /root/catkin_ws
# RUN pip install rospkg
# # RUN catkin_make
# RUN echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
ENV ROS_MASTER_URI=http://172.17.0.1:11311
ENV ROS_IP=172.17.0.1
ENV DISPLAY=unix$DISPLAY
