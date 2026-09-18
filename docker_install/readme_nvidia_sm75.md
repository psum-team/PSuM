[English](readme_nvidia_sm75_en.md) | 中文

### Nvidia-RTX20 系列-SM75

适用于拥有显卡并希望利用 GPU 加速的用户。
sm-75 是指硬件平台的架构号，适用于 Nvidia RTX20 系列显卡。
为了正确运行程序，宿主机必须安装 NVIDIA Driver，且版本需支持 CUDA 12.4。
在终端中执行以下命令可以检查 GPU 驱动是否就绪：

```bash
nvidia-smi
```

- **step 0 拉取 docker base image**
对于国内用户，建议先从国内镜像站获取 base image:
```bash
sudo docker pull docker.m.daocloud.io/nvidia/cuda:12.4.1-devel-ubuntu22.04
sudo docker tag docker.m.daocloud.io/nvidia/cuda:12.4.1-devel-ubuntu22.04 nvidia/cuda:12.4.1-devel-ubuntu22.04
```

- **step 1 开启 CUDA 和配置 SM**

  在**项目根目录**下(不是 docker_install 文件夹内)创建 config.mk.local. CUDA 默认为开启, 因此只需要配置 SM：
```Bash
rm -f config.mk.local
echo "CUDA_ARCH := sm_75" >> config.mk.local
```
- **step 2 构建 docker image**

```bash
# a. 使用清华镜像源（推荐国内用户）：
docker build -t psum:nvidia-sm75 \
  --build-arg MIRROR_URL=//mirrors.tuna.tsinghua.edu.cn \
  -f docker_install/dockerfile_cuda_12_4 .
```

```bash
# b. 使用官方源：
docker build -t psum:nvidia-sm75 -f docker_install/dockerfile_cuda_12_4 .
```
- **step 3 启动容器**

**必须**添加 --gpus all 参数以挂载物理显卡：
```Bash
docker run -it --gpus all --name psum_gpu_dev -v $(pwd):/workspace psum:nvidia-sm75 bash
```
- **step 4 容器内编译**

  进入容器终端后，执行以下标准流程：
```Bash
bash env_scan.sh
source env_load.sh
bash build.sh
```

**编译结果**：
- Field solver: CPU ✅ | Multigrid ✅ | Direct ✅ | GPU ✅
- Pypsum 序列化: ✅ 编译成功 | ✅ Python 交互测试通过

  由于使用了 -v $(pwd):/workspace，在宿主机用 VS Code 修改的代码，
  在容器也能同步看到更改，make 或执行测试会相应生效。

配置后，在**容器内部**应当能够正确编译和运行 psum 中的代码。

最后，将**项目文件夹**下的`env_load.sh`的内容添加到容器内的 `~/.bashrc` 文件中，以便进入容器时能自动加载环境变量。

**后续**再次进入容器：
```bash
docker start -i psum_gpu_dev

# 在容器内加载环境变量
source env_load.sh
```