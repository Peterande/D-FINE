FROM registry.cn-hangzhou.aliyuncs.com/peterande/dfine:v1
RUN python3 -m pip install torch>=2.4.1 torchvision
# docker login --username=xxx registry.cn-hangzhou.aliyuncs.com
# cd [PATH_2_Dockerfile]
# docker build -t xxx:v1 .
# docker tag xxx:v1 registry.cn-hangzhou.aliyuncs.com/xxx/xxx:v1
# docker push registry.cn-hangzhou.aliyuncs.com/xxx/xxx:v1

# FROM dockerpull.com/nvidia/cuda:12.0.1-cudnn8-devel-ubuntu18.04
# ARG DEBIAN_FRONTEND=noninteractive                       
# ENV PATH="/root/miniconda3/bin:${PATH}"                   
# ARG PATH="/root/miniconda3/bin:${PATH}"

# RUN sed -i "s/archive.ubuntu./mirrors.aliyun./g" /etc/apt/sources.list
# RUN sed -i "s/deb.debian.org/mirrors.aliyun.com/g" /etc/apt/sources.list
# RUN sed -i "s/security.debian.org/mirrors.aliyun.com\/debian-security/g" /etc/apt/sources.list
# RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list

# RUN apt-get update && apt-get install -y --no-install-recommends apt-utils && \
#         apt-get upgrade -y && \
#         apt-get install -y vim git libgl1-mesa-glx libglib2.0-0 libsm6 && \
#         apt-get install -y libxrender1 libxext6 tmux wget htop && \
#         apt-get install -y build-essential gcc g++ gdb binutils pciutils net-tools iputils-ping iproute2 git vim wget curl make openssh-server openssh-client tmux tree man unzip unrar

# ENV PYTHONIOENCODING=UTF-8

# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
#         mkdir /root/.conda  && \
#         bash Miniconda3-latest-Linux-x86_64.sh -b  && \
#         rm -f Miniconda3-latest-Linux-x86_64.sh  && \   
#         conda init bash    

# RUN conda config --set show_channel_urls yes \
#         && echo "channels:" > ~/.condarc \
#         && echo " - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/" >> ~/.condarc \
#         && echo " - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/" >> ~/.condarc \ 
#         && echo "show_channel_urls: true" \ 
#         && cat ~/.condarc \
#         && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
#         && cat ~/.config/pip/pip.conf

# RUN python3 -m pip install --upgrade pip  && \
#         python3 -m pip install --upgrade setuptools
        
# RUN python3 -m pip install jupyterlab pycocotools PyYAML tensorboard scipy      
# RUN python3 -m pip --default-timeout=10000 install torch torchvision