English | [中文](readme_no_gpu.md)

### no-gpu Deployment

Suitable for users without a GPU or who only need logic testing.

- **step 0 Pull the Docker base image**

For users in China, it is recommended to pull the base image from a domestic mirror first:
```bash
docker pull docker.m.daocloud.io/library/ubuntu:24.04
docker tag docker.m.daocloud.io/library/ubuntu:24.04 ubuntu:24.04
```

- **step 1 Disable CUDA**

  Create a config.mk.local file in the **project root directory** (not inside the docker_install folder) to disable CUDA:
```Bash
rm -f config.mk.local
echo "USE_CUDA := 0" > config.mk.local
```
- **step 2 Build the Docker image**

```bash
# a. Using Tsinghua mirror (recommended for users in China):
docker build -t psum:no-gpu \
  --build-arg MIRROR_URL=//mirrors.tuna.tsinghua.edu.cn \
  -f docker_install/dockerfile_no_gpu .
```

```bash
# b. Using the official source:
docker build -t psum:no-gpu -f docker_install/dockerfile_no_gpu .
```
- **step 3 Start the container**

  Run the following command in the project root directory to enter the container. Note that the -v flag maps your current source directory to the container's /workspace:
```Bash
docker run -it --name psum_cpu_dev -v $(pwd):/workspace psum:no-gpu bash
```
- **step 4 Compile inside the container**

  After entering the container terminal, execute the following standard procedure:
```Bash
bash env_scan.sh
source env_load.sh
bash build.sh
```

**Build results**:
- Field solver: CPU ✅ | Multigrid ✅ | Direct ✅ | GPU ❌ (no-gpu version)
- Pypsum serialization: ✅ Build succeeded | ✅ Python interaction test passed

  Since -v $(pwd):/workspace is used, code modifications made via VS Code on the host machine will be synchronized inside the container, and make or test executions will take effect accordingly.

After configuration, you should be able to correctly compile and run PSuM code **inside the container**.

Finally, add the contents of `env_load.sh` from the **project folder** to the `~/.bashrc` file inside the container, so that environment variables are automatically loaded when entering the container.

**Later**, to re-enter the container:
```bash
docker start -i psum_cpu_dev
```
