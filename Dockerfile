FROM nvcr.io/nvidia/pytorch:23.11-py3

#### The command below is no longer required.
#### Source of the command below https://askubuntu.com/questions/909277/avoiding-user-interaction-with-tzdata-when-installing-certbot-in-a-docker-contai 
# ENV DEBIAN_FRONTEND=noninteractive 
#### The equivalent for the above command in bash is "export DEBIAN_FRONTEND=noninteractive"

# Set the TZ environment variable to the desired time zone
ENV TZ=America/New_York
#### The equivalent for the above command in bash is "export TZ=America/New_York"

# Fix: tzdata hangs during Docker image build --> https://dev.to/grigorkh/fix-tzdata-hangs-during-docker-image-build-4o9m
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

########################
#### The instructions below are necessary to download jupyter notebooks in PDF format via Latex
#### Instructions from https://tex.stackexchange.com/questions/179778/xelatex-under-ubuntu
########################
RUN apt-get update
RUN apt-get install -y texlive-xetex
RUN apt-get install -y pandoc

#### If I use the "apt-get update" command, I may face conflicts with the current NVIDIA installations in the source image. 
#### Install necessary dependencies
#RUN apt-get update && apt-get install -y \
#    python3-opencv \
#    jupyter-notebook


### https://medium.com/analytics-vidhya/how-to-install-jupyter-notebook-using-pip-e597b5038bb1
#RUN pip install notebook

#RUN apt-get install -y python3-opencv

########################
#### OPENCV INSTALLATION
#### https://docs.opencv.org/3.4/d2/de6/tutorial_py_setup_in_ubuntu.html
########################

### Required build dependencies - OPENCV
RUN apt-get update
RUN apt-get install -y cmake
RUN apt-get install -y gcc g++
RUN apt-get install -y python3-dev python3-numpy
RUN apt-get install -y libavcodec-dev libavformat-dev libswscale-dev 
RUN apt-get install -y libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
RUN apt-get install -y libgtk-3-dev

### Optional Dependencies - OPENCV
RUN apt-get install -y libpng-dev
RUN apt-get install -y libjpeg-dev
RUN apt-get install -y libopenexr-dev
RUN apt-get install -y libtiff-dev
RUN apt-get install -y libwebp-dev

################################################
################################################ MODIFY FROM HERE
###
# Set working directory to the shared host directory
WORKDIR /workspace
RUN git clone https://github.com/opencv/opencv.git
RUN git clone https://github.com/opencv/opencv_contrib.git
WORKDIR /workspace/opencv
RUN mkdir build
WORKDIR /workspace/opencv/build
### The code below was based on https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D WITH_TBB=ON \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D WITH_CUBLAS=1 \
-D WITH_CUDA=ON \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
-D OPENCV_ENABLE_NONFREE=ON \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules ..
RUN make -j10
RUN make install
###
################################################
################################################


### Install Jupyter Notebook
#RUN pip install notebook



# Expose port 8888 for Jupyter Notebook
EXPOSE 8888

# Set working directory to the shared host directory
WORKDIR /workspace

# It is not Possible to mount a host folder using dockerfile. We must use a docker compose file for that.
#### VOLUME ~/Workspace/Computer_Vision_Udacity/Notebooks/P1_Facial_Keypoints:/workspace

# Start Jupyter Notebook server
#CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
#CMD ["/bin/bash"]
#CMD ["nvidia-smi"]
