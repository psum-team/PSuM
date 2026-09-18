English | [中文](readme_nvidia_sm75.md)

### Nvidia-RTX20-Series-SM75

Suitable for users with a GPU who want to leverage GPU acceleration.
sm-75 refers to the hardware platform architecture number, applicable to Nvidia RTX 20 series GPUs.
To run the program correctly, the host machine must have the NVIDIA Driver installed, with a version that supports CUDA 12.4.
Run the following command in the terminal to check if the GPU driver is ready:

```bash
nvidia-smi
```

- **step 0 Pull the Docker base image**

For users in China, it is recommended to pull the base image from a domestic mirror first:
```bash
sudo docker pull docker.m.daocloud.io/nvidia/cuda:12.4.1-devel-ubuntu22.04
sudo docker tag docker.m.daocloud.io/nvidia/cuda:12.4.1-devel-ubuntu22.04 nvidia/cuda:12.4.1-devel-ubuntu22.04
```

- **step 1 Enable CUDA and configure SM**

  Create a config.mk.local file in the **project root directory** (not inside the docker_install folder). CUDA is enabled by default, so you only need to configure SM:
```Bash
rm -f config.mk.local
echo "CUDA_ARCH := sm_75" >> config.mk.local
```
- **step 2 Build the Docker image**

```bash
# a. Using Tsinghua mirror (recommended for users in China):
docker build -t psum:nvidia-sm75 \
  --build-arg MIRROR_URL=//mirrors.tuna.tsinghua.edu.cn \
  -f docker_install/dockerfile_cuda_12_4 .
```

```bash
# b. Using the official source:
docker build -t psum:nvidia-sm75 -f docker_install/dockerfile_cuda_12_4 .
```
- **step 3 Start the container**

**Must** add the --gpus all flag to mount the physical GPU:
```Bash
docker run -it --gpus all --name psum_gpu_dev -v $(pwd):/workspace psum:nvidia-sm75 bash
```
- **step 4 Compile inside the container**

  After entering the container terminal, execute the following standard procedure:
```Bash
bash env_scan.sh
source env_load.sh
bash build.sh
```

**Build results**:
- Field solver: CPU ✅ | Multigrid ✅ | Direct ✅ | GPU ✅
- Pypsum serialization: ✅ Build succeeded | ✅ Python interaction test passed

  Since -v $(pwd):/workspace is used, code modifications made via VS Code on the host machine will be synchronized inside the container, and make or test executions will take effect accordingly.

After configuration, you should be able to correctly compile and run PSuM code **inside the container**.

Finally, add the contents of `env_load.sh` from the **project folder** to the `~/.bashrc` file inside the container, so that environment variables are automatically loaded when entering the container.

**Later**, to re-enter the container:
```bash
docker start -i psum_gpu_dev

# Load environment variables inside the container
source env_load.sh
```
