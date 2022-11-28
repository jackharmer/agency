FROM nvidia/cuda:11.3.1-base-ubuntu20.04

# Remove third-party sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    sudo 
RUN rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN curl -sLo /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh \
    && chmod +x /miniconda.sh \
    && /miniconda.sh -b -p /miniconda \
    && rm /miniconda.sh 
ENV PATH /miniconda/bin:$PATH 
RUN conda update -n base -c defaults conda

# Required for building box2d wheel
RUN apt-get update && apt-get install -y \
    swig \
    build-essential 

# Create conda env and install requirements of agency project
COPY conda_env.yaml /conda_env.yaml
RUN conda env create -f /conda_env.yaml \
    && rm /conda_env.yaml \
    && conda clean -ya
ENV PATH /miniconda/envs/agency/bin:$PATH
RUN echo "source activate agency" > ~/.bashrc

RUN mkdir /agency
COPY . /agency
WORKDIR /agency
# Install agency as editable, such that when using code from the host (via the docker -v command) changes are picked up.
RUN python -m pip install -e .

CMD ["bash"]

# - Build:
# docker build -t agency .
#
# - Run the tests, using code from when the image was built:
# docker run --gpus all -it agency python examples/gym/train_sac_mlp_continuous_lunarlander.py
#
# - Run the tests, using code from the current working directory on the host:
# docker run --gpus all -v $(pwd):/agency -it agency python examples/gym/train_sac_mlp_continuous_lunarlander.py
