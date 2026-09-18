English | [中文](readme.md)

> **Image distribution**: images are published to GHCR (`ghcr.io/psum-team/psum_env`,
> tags: `no_gpu`, `cuda_12_4`).
>
> **Note**: these are *environment images* (compiler toolchain and dependencies only,
> no PSuM source inside). PSuM itself lives in this repository and is mounted into
> the container at runtime; updating PSuM does not require a new image. Only
> dependency/toolchain changes trigger an image rebuild.

## Deploy PSuM with Docker

Before starting, make sure Docker is installed. For Windows users, we recommend installing WSL2 first, then Docker Desktop.

[Docker Official Site](https://docs.docker.com/get-started/get-docker/)

### no-gpu Deployment

Suitable for users without a GPU or who only need logic testing.
Navigate to the **project root directory** and enter the following commands in the terminal to disable CUDA:

> ⚠️ The two commands below **delete and recreate** `config.mk.local`. If you already have custom settings, edit the file manually instead and only add/change the `USE_CUDA := 0` line.

```Bash
rm -f config.mk.local
echo "USE_CUDA := 0" > config.mk.local
```
Enter the following in the terminal:
```bash
# 1. Pull the image
sudo docker pull ghcr.io/psum-team/psum_env:no_gpu
# 2. Run the container (execute in the psum project root directory)
sudo docker run -it --name psum_dev -v $(pwd):/workspace ghcr.io/psum-team/psum_env:no_gpu bash
# 3. First compilation inside the container; you should see the build succeed.
./build.sh
# 4. To re-enter the container each time you need to use it:
sudo docker start psum_dev # If the container is not running, use start to launch it
sudo docker exec -it psum_dev bash # Enter the container
```
*If you need to rebuild the Docker image from scratch, refer to [jump to no-gpu](readme_no_gpu_en.md).*



### Deploy using cuda-12.4 Docker image
**Prerequisites**:
- NVIDIA driver installed on the new machine (supporting CUDA 12.4)
- Docker installed on the new machine
- NVIDIA Container Toolkit installed on the new machine (required for --gpus all; WSL2 users can skip this step)

You may need to create a config.mk.local file in the **project root directory** and configure the SM version.

> ⚠️ The two commands below **delete and recreate** `config.mk.local`. If the file already has content (e.g. `USE_CUDA := 0`), edit it manually instead and only add/change the `CUDA_ARCH` line.

```bash
rm -f config.mk.local
echo "CUDA_ARCH := sm_89" >> config.mk.local # Nvidia RTX 20 series: sm75 (default, can skip this step). RTX 30 series: sm86. RTX 40 series: sm89. RTX 50 series: sm120 (needs CUDA 12.8+ toolchain);
```
Enter the following in the terminal:
```bash
# 1. Pull the image
sudo docker pull ghcr.io/psum-team/psum_env:cuda_12_4
# 2. Run the container (execute in the psum project root directory)
sudo docker run -it --gpus all --name psum_dev -v $(pwd):/workspace ghcr.io/psum-team/psum_env:cuda_12_4 bash
# 3. First compilation inside the container; you should see the build succeed.
./build.sh
# 4. To re-enter the container each time you need to use it:
sudo docker start psum_dev # If the container is not running, use start to launch it
sudo docker exec -it psum_dev bash # Enter the container
```
*If you need to rebuild the Docker image from scratch, refer to [jump to nvidia-sm75](readme_nvidia_sm75_en.md).*
