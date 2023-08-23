FROM ubuntu:20.04
RUN apt-get update

RUN echo 'root:root' | chpasswd
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server rsync vim python3-pip
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y sudo git python3-pip libgl1-mesa-dev
RUN sed -i 's/\#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/\#PermitEmptyPasswords no/PermitEmptyPasswords yes/' /etc/ssh/sshd_config

RUN \
  useradd h-ishida && \
  echo "h-ishida ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/h-ishida && \
  chmod 0440 /etc/sudoers.d/h-ishida && \
  mkdir -p /home/h-ishida && \
  chown h-ishida:h-ishida /home/h-ishida && \
  chsh -s /bin/bash h-ishida

RUN echo 'root:root' | chpasswd
RUN echo 'h-ishida:h-ishida' | chpasswd
WORKDIR /home/h-ishida
USER h-ishida

# copy ssh
COPY --chown=h-ishida .ssh /home/h-ishida/.ssh

RUN sudo apt-get update && sudo apt install ffmpeg -y

# skrobot stuff (must build some deps)
RUN sudo apt update && sudo apt install cmake -y
RUN sudo apt install liblapack-dev libopenblas-dev libsuitesparse-dev -y
RUN export CPPFLAGS="-I/usr/include/suitesparse" && pip3 install cvxopt -v
RUN sudo apt-get install libfcl-dev liboctomap-dev python3-numpy -y
# unfortunately, python-fcl cannot be compiled with recent cython
RUN pip3 install "cython<3.0"
RUN pip3 install python-fcl -v
RUN sudo apt remove python3-numpy -y  # this cause skbuild error somehow

RUN \
    git clone git@github.com:iory/pySDFGen.git && \
    cd pySDFGen && \
    git submodule update --init && \
    pip3 install -e .
RUN \
    git clone git@github.com:iory/scikit-robot.git && \
    cd scikit-robot && \
    pip3 install -e .

# tinyfk
RUN sudo apt-get install libeigen3-dev -y  # tinyfk
RUN pip3 install tinyfk -v

# voxbloxpy
RUN sudo apt-get install libgoogle-glog-dev -y  # voxbloxpy
RUN pip3 install voxbloxpy

# ompl-python-thin
RUN sudo apt-get install libboost-all-dev -y  # ompl
RUN pip3 install ompl-thin

RUN git clone git@github.com:HiroIshida/scikit-motionplan.git && pip3 install -e scikit-motionplan
RUN git clone git@github.com:HiroIshida/rpbench.git && pip3 install -e rpbench
RUN git clone git@github.com:HiroIshida/hifuku.git && cd hifuku && pip3 install -e .
RUN pip3 install pyglet==1.5.27

# finally install required to run skrobot
RUN sudo apt-get install libspatialindex-dev freeglut3-dev libsuitesparse-dev libblas-dev liblapack-dev -y
