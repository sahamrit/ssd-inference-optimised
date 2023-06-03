FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu116-py38-torch1121

# Install pip dependencies
RUN pip install 'ipykernel~=6.20.2' \
                'azureml-core==1.50.0' \
				'azureml-dataset-runtime==1.50.0' \
                'azureml-defaults==1.50.0' \
				'azure-ml==0.0.1' \
				'azure-ml-component==0.9.18' \
                'azureml-mlflow==1.50.0' \
                'azureml-telemetry==1.50.0' \
		        'azureml-contrib-services==1.50.0' \
                'torch-tb-profiler~=0.4.0' \
				'py-spy==0.3.12' \
		        'debugpy~=1.6.3'

RUN pip install \
        azureml-inference-server-http~=0.8.0 \
        inference-schema~=1.5.0 \
        MarkupSafe==2.1.2 \
	    regex \
	    pybind11

# Inference requirements
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20230419.v1 /artifacts /var/
RUN /var/requirements/install_system_requirements.sh && \
    cp /var/configuration/rsyslog.conf /etc/rsyslog.conf && \
    cp /var/configuration/nginx.conf /etc/nginx/sites-available/app && \
    ln -sf /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
    rm -f /etc/nginx/sites-enabled/default
ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=400
EXPOSE 5001 8883 8888

# support Deepspeed launcher requirement of passwordless ssh login
RUN apt-get update
RUN apt-get install -y openssh-server openssh-client

#install Gstreamer
RUN apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio

RUN  apt install -y python3-gi python3-gi-cairo gir1.2-gtk-3.0 libgirepository1.0-dev gcc libcairo2-dev pkg-config python3-dev

ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7

RUN pip install 'pycairo' \
                'PyGObject' \
                'nvtx' \
                'scikit-image' \
                'gdown' \
                'matplotlib' 

RUN apt install -y libssl1.1 libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstrtspserver-1.0-0 libjansson4 libyaml-cpp-dev gcc make git python3

# Install TensorRT and DeepStream
RUN gdown https://drive.google.com/uc?id=1U7-fAg-qxmBVp2od9L0Sc1d9017_JUqj && \
    gdown https://drive.google.com/uc?id=1xBnRqHuNzYSEmbjbzWIw7YaB-VE4Keok

RUN rm /etc/apt/sources.list.d/*cuda* && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" | sudo tee /etc/apt/sources.list.d/cuda-repo.list && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    apt-key add 3bf863cc.pub && \
    apt-get update

RUN dpkg -i nv-tensorrt-repo-ubuntu2004-cuda11.4-trt8.2.5.1-ga-20220505_1-1_amd64.deb && \
    apt-key add /var/nv-tensorrt-repo-ubuntu2004-cuda11.4-trt8.2.5.1-ga-20220505/82307095.pub && \
    apt-get update && \
    apt install -y tensorrt

RUN git clone https://github.com/edenhill/librdkafka.git && \
    cd librdkafka && \
    git reset --hard 7101c2310341ab3f4675fc565f64f0967e135a6a && \ 
    ./configure && \
    make && \
    make install

RUN cd .. && \
    mkdir -p /opt/nvidia/deepstream/deepstream-6.1/lib && \
    cp /usr/local/lib/librdkafka* /opt/nvidia/deepstream/deepstream-6.1/lib

RUN apt-get install ./deepstream-6.1_6.1.0-1_amd64.deb











