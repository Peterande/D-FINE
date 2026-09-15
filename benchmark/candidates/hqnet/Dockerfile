FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential git libglib2.0-0 libgl1 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html \
      mmcv-full==1.7.0 \
 && pip install --no-cache-dir \
      mmdet==2.25.3 numpy==1.23.5 scipy==1.10.1 opencv-python-headless==4.8.1.78 \
      einops==0.7.0 munkres==1.1.4 terminaltables==3.1.10 pycocotools==2.0.7

WORKDIR /work
