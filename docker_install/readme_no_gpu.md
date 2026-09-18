[English](readme_no_gpu_en.md) | 中文

### no-gpu 部署

适用于没有显卡或仅需进行逻辑测试的用户。

- **step 0 拉取 docker base image**
对于国内用户，建议先从国内镜像站获取 base image:
```bash
docker pull docker.m.daocloud.io/library/ubuntu:24.04
docker tag docker.m.daocloud.io/library/ubuntu:24.04 ubuntu:24.04
```

- **step 1 关闭 CUDA**

  在**项目根目录**下(不是 docker_install 文件夹内)创建 config.mk.local，禁用 CUDA：
```Bash
rm -f config.mk.local
echo "USE_CUDA := 0" > config.mk.local
```
- **step 2 构建 docker image**

```bash
# a. 使用清华镜像源（推荐国内用户）：
docker build -t psum:no-gpu \
  --build-arg MIRROR_URL=//mirrors.tuna.tsinghua.edu.cn \
  -f docker_install/dockerfile_no_gpu .
```

```bash
# b. 使用官方源：
docker build -t psum:no-gpu -f docker_install/dockerfile_no_gpu .
```
- **step 3 启动容器**

  在项目根目录下执行以下命令进入容器。注意 -v 参数会将你当前的源码目录映射到容器的 /workspace：
```Bash
docker run -it --name psum_cpu_dev -v $(pwd):/workspace psum:no-gpu bash
```
- **step 4 容器内编译**

  进入容器终端后，执行以下标准流程：
```Bash
bash env_scan.sh
source env_load.sh
bash build.sh
```

**编译结果**：
- Field solver: CPU ✅ | Multigrid ✅ | Direct ✅ | GPU ❌ (no-gpu 版本)
- Pypsum 序列化: ✅ 编译成功 | ✅ Python 交互测试通过

  由于使用了 -v $(pwd):/workspace，在宿主机用 VS Code 修改的代码，
  在容器也能同步看到更改，make 或执行测试会相应生效。

配置后，在**容器内部**应当能够正确编译和运行 psum 中的代码。

最后，将**项目文件夹**下的`env_load.sh`的内容添加到容器内的 `~/.bashrc` 文件中，以便进入容器时能自动加载环境变量。

**后续**再次进入容器：
```bash
docker start -i psum_cpu_dev
```