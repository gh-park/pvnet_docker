FROM nvidia/cudagl:10.0-devel-ubuntu18.04
MAINTAINER minchang <tjdalsckd@gmail.com>
RUN gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv A4B469963BF863CC
RUN gpg --export --armor A4B469963BF863CC | apt-key add -
RUN apt-get update &&  apt-get install -y -qq --no-install-recommends \
    libgl1 \
    libxext6 \ 
    libx11-6 \
   && rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute


RUN echo 'export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}' >> ~/.bashrc
RUN echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
RUN echo 'export PATH=/usr/local/cuda/bin:/$PATH' >> ~/.bashrc
RUN echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils

RUN apt-get install -y wget
RUN apt-get install -y sudo curl
RUN su root
RUN apt-get install -y python
RUN apt-get update && apt-get install -y lsb-release && apt-get clean all
RUN  apt-get -yq update && \
     apt-get -yqq install ssh
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

ARG ssh_prv_key
ARG ssh_pub_key
RUN mkdir -p ~/.ssh && \
    chmod 0700 ~/.ssh && \
    ssh-keyscan github.com > ~/.ssh/known_hosts
RUN echo "$ssh_prv_key" > ~/.ssh/id_rsa && \
    echo "$ssh_pub_key" > ~/.ssh/id_rsa.pub && \
    chmod 600 ~/.ssh/id_rsa && \
    chmod 600 ~/.ssh/id_rsa.pub
RUN apt-get install -y git
RUN apt-get install -y gedit
RUN echo "conda activate pvnet" >> ~/.bashrc

SHELL ["/bin/bash", "--login", "-c"]

ENV PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN conda update -n base -c defaults conda
RUN conda create -n pvnet python=3.7
RUN conda init bash
RUN conda activate pvnet
RUN wget https://download.pytorch.org/whl/cu100/torch-1.3.0%2Bcu100-cp37-cp37m-linux_x86_64.whl
RUN conda activate pvnet;conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
RUN conda activate pvnet;pip install Cython==0.28.2
RUN apt-get install -y libglfw3-dev libglfw3
RUN mkdir -p /root/workspace;cd /root/workspace/; git clone git@github.com:tjdalsckd/pvnet_smc.git;
RUN  conda activate pvnet ; pip install yacs==0.1.4 numpy==1.16.4 torchvision==0.2.1 opencv-python==3.4.2.17 tqdm==4.28.1 pycocotools==2.0.0 matplotlib==2.2.2 plyfile==0.6 scikit-image==0.14.2 scikit-learn PyOpenGL==3.1.1a1 ipdb==0.13.9 cyglfw3==3.1.0.2 pyassimp==3.3 progressbar==2.5 open3d-python==0.5.0.0 tensorboardX==1.2 cffi==1.11.5
ENV ROOT=/root/workspace/pvnet_smc
ENV CUDA_VISIBLE_DEVICES=0,1
RUN conda activate pvnet ;cd $ROOT/lib/csrc;export CUDA_HOME="/usr/local/cuda";cd dcn_v2;python setup.py build_ext --inplace;cd ../ransac_voting;python setup.py build_ext --inplace;cd ../fps;python setup.py; 
RUN pip install gdown

RUN gdown  https://drive.google.com/uc?id=1ncA2-SAsNkIQsYWhehX5rai9Z1IkSlyT
RUN gdown  https://drive.google.com/uc?id=1Ag2fMrfdzwdH82YEb6TBkjXeEmlMoiA8
RUN gdown  https://drive.google.com/uc?id=1Lp_iAWjgzOeyOobV8f7SvRUJ61vC0weO
RUN gdown  https://drive.google.com/uc?id=1MvsVqO5widvW3E95aFNbEFOjmo4JTdDp

RUN mv TruncationLINEMOD.tar.gz /root/workspace/
RUN mv OCCLUSION_LINEMOD.tar.gz /root/workspace/
RUN mv LINEMOD_ORIG.tar.gz /root/workspace/
RUN mv LINEMOD.tar.gz /root/workspace/
RUN cd /root/workspace/; find . -name '*tar.gz' -exec tar xvf {} \;
RUN cd $ROOT/data;ln -s /root/workspace/LINEMOD linemod;ln -s /root/workspace/LINEMOD_ORIG linemod_orig;ln -s /root/workspace/OCCLUSION_LINEMOD occlusion_linemod
RUN cd /root/workspace/;wget https://download.blender.org/release/Blender2.79/blender-2.79a-linux-glibc219-x86_64.tar.bz2; tar -xvf blender-2.79a-linux-glibc219-x86_64.tar.bz2;mv blender-2.79a-linux-glibc219-x86_64 blender
RUN cd /root/workspace/; git clone git@github.com:tjdalsckd/pvnet-rendering-smc.git
RUN wget http://groups.csail.mit.edu/vision/SUN/releases/SUN2012pascalformat.tar.gz
RUN mv SUN2012pascalformat.tar.gz /root/workspace/
RUN cd /root/workspace/;tar -xvf SUN2012pascalformat.tar.gz
RUN apt-get install libglu1
RUN cd /root/workspace/;rm -r pvnet-rendering-smc;ls -a;git clone git@github.com:tjdalsckd/pvnet-rendering-smc.git;
RUN conda activate pvnet;pip install opencv-python
RUN apt update && apt install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev
RUN apt-get install libsm6 libxrender1 libfontconfig1
RUN conda activate pvnet; pip install opencv-contrib-python
RUN conda activate pvnet; pip install easydict
RUN conda activate pvnet; pip install lmdb
RUN conda activate pvnet;  pip install transforms3d;
RUN apt-get install -y openexr
RUN apt-get install -y libopenexr-dev
RUN conda activate pvnet; pip install OpenEXR --user
RUN apt-get install -y nautilus;
RUN cp /root/workspace/pvnet-rendering-smc/get-pip.py /root/workspace/blender/2.79/python/bin/
RUN cd /root/workspace/blender/2.79/python/bin/; ./python3.5m get-pip.py;./python3.5m -m pip install --upgrade pip;./python3.5m -m pip uninstall numpy; ./python3.5m -m pip install numpy==1.15.4;./pip install transforms3d;./python3.5m -m pip install easydict
RUN cd /root/workspace/pvnet-rendering-smc; mkdir -p data; mv /root/workspace/SUN2012pascalformat data/SUN
RUN mkdir -p /home/vision/workspace/pvnet-rendering-smc/data/; cd /home/vision/workspace/pvnet-rendering-smc/data/; ln -s /root/workspace/pvnet-rendering-smc/data/SUN .
RUN cp -r /root/workspace/pvnet-rendering-smc/data/LINEMOD/* /root/workspace/pvnet_smc/data/linemod/
RUN rm -r /root/workspace/pvnet-rendering-smc/data/LINEMOD/
RUN rm -r /root/workspace/pvnet-rendering-smc/data/LINEMOD_ORIG/

RUN cd /root/workspace/pvnet-rendering-smc/data/; ln -s /root/workspace/pvnet_smc/data/linemod LINEMOD;
RUN cd /root/workspace/pvnet-rendering-smc/data/;ln -s /root/workspace/LINEMOD LINEMOD;ln -s /root/workspace/LINEMOD_ORIG LINEMOD_ORIG;ln -s /root/workspace/OCCLUSION_LINEMOD OCCLUSION_LINEMOD
RUN rm -r /root/workspace/pvnet-rendering-smc/data/LINEMOD/renders/cat/
RUN apt-get install -y imagemagick
RUN echo 'conda activate pvnet;cd /root/workspace'>>~/.bashrc
RUN echo '#!/bin/bash'>>/root/workspace/make_dataset.sh
RUN echo 'cd /root/workspace/pvnet-rendering-smc;bash move_file.sh'>>/root/workspace/make_dataset.sh
RUN chmod 777 /root/workspace/make_dataset.sh
RUN echo '#!/bin/bash'>>/root/workspace/start_training.sh
RUN echo 'cd /root/workspace/pvnet_smc/data;ln -s /root/workspace/pvnet-rendering-smc/custom .;pip install numpy --upgrade;cd /root/workspace/pvnet_smc; python run.py --type custom;python train_net.py --cfg_file configs/linemod.yaml train.dataset CustomTrain test.dataset CustomTrain model mycat train.batch_size 4;'>>/root/workspace/start_training.sh

RUN cd /root/workspace/;ls; git clone git@github.com:tjdalsckd/pvnet-rendering-smc.git pvnet-rendering-smc3;
RUN cd /root/workspace/ ;cp -r pvnet-rendering-smc3 pvnet-rendering-smc; rm -r pvnet-rendering-smc3;
RUN echo '#!/bin/bash'>>/root/workspace/test.sh
RUN echo 'cd /root/workspace/pvnet_smc;python3 run.py --type visualize --cfg_file configs/linemod.yaml train.dataset CustomTrain test.dataset CustomTrain model mycat'>>/root/workspace/test.sh
RUN chmod 777 /root/workspace/*.sh
RUN conda activate pvnet;cd /root/workspace/pvnet_smc;git clone https://github.com/tjdalsckd/calibration_docker;  git clone https://github.com/tjdalsckd/gqcnnddddd gqcnn; 
RUN conda activate pvnet;cd /root/workspace/pvnet_smc/gqcnn/gqcnn;pip3 install -U pip;pip3 install trimesh;cd gqcnn; git clone https://github.com/BerkeleyAutomation/meshrender.git; pip3 install autolab-core==0.0.14 autolab-perception==0.0.8 visualization==0.1.1;
RUN conda activate pvnet; cd /root/workspace/pvnet_smc/gqcnn/gqcnn;pip3 install .
RUN conda activate pvnet; pip3 install numpy==1.16.4; pip3 install pyrealsense2;
RUN apt-get install unzip
RUN conda activate pvnet; cd /root/workspace/pvnet_smc/calibration_docker; unzip GQCNN-4.0-PJ
RUN conda activate pvnet; cd /root/workspace/pvnet_smc; mkdir -p models; cd models; cp -r ../calibration_docker/GQCNN* .

RUN echo 'pip3 install numpy==1.16.4'>>/root/workspace/make_dataset.sh
RUN echo '#!/bin/bash'>>/root/workspace/start_realsense.sh
RUN echo 'pip3 install numpy --upgrade;cd /root/workspace/pvnet_smc;python3 test_realsense.py --type visualize --cfg_file configs/linemod.yaml train.dataset CustomTrain test.dataset CustomTrain model mycat'>>/root/workspace/start_realsense.sh
RUN cd /root/workspace/;chmod 777 start_realsense.sh

