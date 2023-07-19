FROM pytorch/pytorch:latest

# Args -tka
ARG user=przybilla
ARG userid=1001

RUN apt-get -y update
RUN apt-get -y install openslide-tools
RUN apt-get -y install python3-openslide
RUN apt-get -y install libgl1-mesa-glx
RUN apt-get -y install git
RUN apt-get -y install build-essential

# install mothi -tka
RUN pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple mothi

RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install seaborn
RUN pip install openslide-python
RUN pip install opencv-python
RUN pip install aicspylibczi
RUN pip install albumentations
RUN pip install scikit-learn
RUN pip install notebook
RUN pip install torchviz

RUN pip install progress
RUN pip install tensorflow
RUN pip install ptitprince

#TODO
RUN pip install pickle5

RUN pip install torchmetrics
RUN pip install mxnet
RUN pip install pytorch-msssim
RUN pip install paquo
RUN pip install shapely

WORKDIR /METHnet_GBM_segmentation

# install QuPath 0.3.2 and set the enviroment variable -tka
RUN python -m paquo get_qupath --install-path ./ 0.3.2
ENV PAQUO_QUPATH_DIR=/METHnet_GBM_segmentation/QuPath-0.3.2

COPY . /METHnet_GBM_segmentation

# install local mothi version
RUN pip install ./GBM_QuPath_tiles-master

# # set local user... otherwise you can not acces the QuPath project outside of Docker -tka
# RUN useradd -u ${userid} ${user}
# USER ${user}