[English](readme_en.md) | 中文

## 利用 Docker 部署 PSuM

> 镜像分发：中文文档使用阿里云镜像源（国内网络友好）；英文文档使用 GHCR（ghcr.io/psum-team）。两个源提供相同环境的镜像。
>
> 说明：这里是**环境镜像**（仅编译器工具链与依赖库，不含 PSuM 源码）。PSuM 代码始终在本仓库中，
> 运行时挂载进容器即可；PSuM 更新无需重新发布镜像，仅当依赖环境变化时才需要重建发布。

开始前，请确保已经安装了 Docker。对于 windows 用户，我们建议先安装 wsl2，然后安装 docker desktop。

[docker 官网](https://docs.docker.com/get-started/get-docker/)

### no-gpu 部署

适用于没有显卡或仅需进行逻辑测试的用户。
进入**项目根目录**后在终端中输入下面指令以关闭 cuda.

> ⚠️ 下面两条命令会**删除并重建** `config.mk.local`。若你已有自定义配置，请改为手动编辑该文件，仅添加/修改 `USE_CUDA := 0` 一行。

```Bash
rm -f config.mk.local
echo "USE_CUDA := 0" > config.mk.local
```
在终端中输入：
```bash
# 1. 拉取镜像
sudo docker pull crpi-vy93epqqxgjguf3h.cn-shanghai.personal.cr.aliyuncs.com/psum_env_docker/docker_images:no_gpu
# 2. 运行容器 (在 psum 项目根目录执行)
sudo docker run -it --name psum_dev -v $(pwd):/workspace crpi-vy93epqqxgjguf3h.cn-shanghai.personal.cr.aliyuncs.com/psum_env_docker/docker_images:no_gpu bash
# 3. 容器内首次编译; 应当看到构建正确.
./build.sh
# 4. 后续每次需要使用时，进入容器：
sudo docker start psum_dev # 如果容器没有启动，需要start来启动
sudo docker exec -it psum_dev bash # 进入容器
```
*如果需要从新构建 docker image，可参考[jump to no-gpu](readme_no_gpu.md).*



### 使用 cuda-12.4 版本的 docker image 部署
**前提条件**：
- 新机器已安装 NVIDIA 驱动（支持 CUDA 12.4）
- 新机器已安装 Docker
- 新机器已安装 NVIDIA Container Toolkit（才能使用 --gpus all; wsl2 用户无需这一步）
可能需要在**项目根目录**下新建 config.mk.local 并配置 sm 版本.

> ⚠️ 下面两条命令会**删除并重建** `config.mk.local`。若该文件已有内容（例如 `USE_CUDA := 0`），请改为手动编辑，仅添加/修改 `CUDA_ARCH` 一行。

```bash
rm -f config.mk.local
echo "CUDA_ARCH := sm_89" >> config.mk.local # Nvidia RTX 20系列: sm75(默认, 可跳过此步骤). RTX 30系列: sm86. RTX 40系列: sm89. RTX 50系列: sm120(需 CUDA 12.8 及以上工具链);
```
在终端中输入：
```bash
# 1. 拉取镜像
sudo docker pull crpi-vy93epqqxgjguf3h.cn-shanghai.personal.cr.aliyuncs.com/psum_env_docker/docker_images:cuda_12_4
# 2. 运行容器 (在 psum 项目根目录执行)
sudo docker run -it --gpus all --name psum_dev -v $(pwd):/workspace crpi-vy93epqqxgjguf3h.cn-shanghai.personal.cr.aliyuncs.com/psum_env_docker/docker_images:cuda_12_4 bash
# 3. 容器内首次编译; 应当看到构建正确.
./build.sh
# 4. 后续每次需要使用时，进入容器：
sudo docker start psum_dev # 如果容器没有启动，需要start来启动
sudo docker exec -it psum_dev bash # 进入容器
```
*如果需要从新构建 docker image，可参考[jump to nvidia-sm75](readme_nvidia_sm75.md).*