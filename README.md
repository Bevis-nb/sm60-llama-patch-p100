# Tesla P100 (sm_60) llama.cpp 编译与优化指南

本项目基于 `llama.cpp v0.5.0`，针对 NVIDIA Tesla P100 (Pascal 架构，算力 sm_60) 进行源码编译与社区性能补丁（`shinbunbun/llama-cpp-p100-patches`）的应用集成。

> 
> 💡 **背景**：P100 缺乏现代 GPU 的 INT8/INT4 硬件加速指令（如 DP4A），且官方预编译版本通常不包含 sm_60 代码。直接运行会导致 GPU 回退到低效计算路径，因此**必须手动编译并显式指定架构**，才能发挥出应有性能。

---

## 📖 目录

1. [环境准备 (Windows)](#1-%E7%8E%AF%E5%A2%83%E5%87%86%E5%A4%87-windows)
2. [获取源码与打补丁](#2-%E8%8E%B7%E5%8F%96%E6%BA%90%E7%A0%81%E4%B8%8E%E6%89%93%E8%A1%A5%E4%B8%81)
3. [CMake 配置与编译](#3-cmake-%E9%85%8D%E7%BD%AE%E4%B8%8E%E7%BC%96%E8%AF%91)
4. [启动脚本与环境变量](#4-%E5%90%AF%E5%8A%A8%E8%84%9A%E6%9C%AC%E4%B8%8E%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F)
5. [性能提示与排错](#5-%E6%80%A7%E8%83%BD%E6%8F%90%E7%A4%BA%E4%B8%8E%E6%8E%92%E9%94%99)

---

## 1. 环境准备 (Windows)

### 安装 CMake

以管理员身份打开 PowerShell 或 CMD，执行：

```
winget install --scope machine --id Kitware.CMake -e --silent --accept-package-agreements --accept-source-agreements --override "/qn /norestart ADD_CMAKE_TO_PATH=System"
```

### 安装 Visual Studio Build Tools

不需要安装完整的 Visual Studio IDE，只需安装 [https://visualstudio.microsoft.com/downloads/](https://visualstudio.microsoft.com/downloads/) 。

运行安装程序，在 “工作负荷” 中勾选 “使用 C++ 的桌面开发”。请确保右侧安装详细信息中保留以下组件：

- MSVC v143 - VS 2022 C++ x64/x86 生成工具（核心编译器 cl.exe）
- Windows 11 SDK（提供系统头文件）
- 用于 Windows 的 C++ CMake 工具（包含 Ninja，保留可加快编译速度）

(注：可以取消勾选 C++ AddressSanitizer 和 vcpkg，编译 llama.cpp 不需要它们。)

## 2. 获取源码与打补丁

补丁集基于 llama.cpp v0.5.0 生成，并保证在该版本上 “零模糊、零偏移” 应用，请严格切换至该版本。

```
# 1. 克隆官方源码并切换基线版本
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout v0.5.0

# 2. 克隆补丁仓库（假设放在 llama.cpp 同级目录）
git clone https://github.com/shinbunbun/llama-cpp-p100-patches.git ../p100-patches
```

### 应用补丁

推荐使用 Git Bash（在开始菜单中搜索打开），进入源码目录执行：

```
cd /f/tools/llama.cpp  # 路径请根据实际盘符调整
for patch in ../p100-patches/patches/*.patch; do
    git apply --check "$patch" && git apply "$patch"
done
```

(注意：如果补丁仓库克隆在 llama.cpp 文件夹内部，请将 `../p100-patches` 改为 `p100-patches`。)

⚠️ 关于 whitespace 警告
终端可能会输出大量类似 `warning: 496 lines add whitespace errors` 的警告。这属于行尾空格警告，不代表应用失败，不影响补丁内容写入源码，可以完全忽略。

### 验证补丁是否生效

执行以下命令查看变更状态：

```
git status
git diff --stat
```

预期能看到几十个文件被修改，且集中在 `ggml-cuda/` 下的内核文件（如 `mmvq.cu`, `fattn-*.cu` 等）以及 `ggml-cuda.cu`。如果改动为空，请使用 `git checkout .` 回滚后重新排查。

## 3. CMake 配置与编译

进入 **x64 Native Tools Command Prompt for VS 2022**（必须使用此终端，确保能调用 MSVC 编译器），进入源码目录执行配置。

推荐使用 Visual Studio 自带的构建系统（MSBuild）：

```
:: 配置 CMake (关键：指定 sm_60)
cmake -B build -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 -DCMAKE_BUILD_TYPE=Release

:: 执行编译 (开启 8 线程加速)
cmake --build build --config Release -j 8
```

`-DCMAKE_CUDA_ARCHITECTURES=60` 是让 P100 性能补丁生效的必要条件，确保 CUDA 内核针对 GP100 原生编译。

编译完成后，可执行文件位于 `build\bin\Release\llama-server.exe`。

🛠️ 常见报错处理：`No CUDA toolset found`
如果 CMake 配置阶段出现此报错，请手动补全缺失的 CUDA 集成文件：

1. 打开 CUDA 安装目录（如 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\extras\visual_studio_integration\MSBuildExtensions`）。
2. 复制该目录下所有 `.props` 和 `.targets` 文件。
3. 粘贴到 VS Build Tools 目录：`C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\BuildCustomizations`（若 v170 文件夹不存在请手动创建）。
4. 删除 build 文件夹（`rmdir /s /q build`），重新运行 CMake 配置命令。

## 4. 启动脚本与环境变量

补丁中的大量优化是通过环境变量显式开启的。根据实测，以下三个变量是释放 P100 极致性能的核心必开项（缺一不可，否则速度会明显受限）。

在 Windows 下（保存为 `start.bat` 运行）：

```
@echo off
chcp 65001 >nul
title llama.cpp Server - P100 Patch

:: ===== P100 补丁版核心环境变量（必须全部开启） =====
set GGML_CUDA_P60_MMVQ=1
set LLAMA_QWEN4EXP_HC_FOLD=1
set GGML_SCHED_SPLIT_INPUTS_CAP=1

:: ================= 核心配置 =================
set CUDA_VISIBLE_DEVICES=0
set MODEL_PATH=.\model\amoral-gpt-oss-20b-gguf-Q4_K_M.gguf
set CONTEXT_SIZE=32768
set PORT=8080

:: 启动服务
.\build\bin\Release\llama-server.exe ^
  -m %MODEL_PATH% ^
  --alias gpt-oss-20b-amoral ^
  -c %CONTEXT_SIZE% ^
  --n-gpu-layers 99 ^
  -t 8 ^
  -b 2048 ^
  -ub 512 ^
  --parallel 1 ^
  --cache-type-k q8_0 ^
  --cache-type-v q8_0 ^
  -fa on ^
  --no-warmup ^
  --host 127.0.0.1 ^
  --port %PORT%

pause
```

(Linux 环境下，将 `set` 替换为 `export`，并使用 `./build/bin/llama-server` 启动即可。)

## 5. 性能提示与排错

- **显存管理**：P100 拥有 16GB 显存，建议将上下文（`-c`）控制在 32768 左右，避免过长的 KV Cache 挤占计算缓冲区。如果出现 OOM（显存溢出），请尝试将 `--n-gpu-layers 99` 降低至 95 或 90，让少数层回退到 CPU 计算。
- **量化格式选择**：P100 对 Q4_K_M 这类较新的 K-quant 量化反量化效率较低。建议优先下载并使用传统的 Q4_0 或 Q4_1 格式，在老架构上表现更好。
- **模型架构瓶颈**：P100 跑 20B 级别的 MoE（混合专家）模型存在先天劣势（如 GPT-OSS）。如果实测速度依然不理想，建议更换为 7B~14B 的 Dense（稠密）模型（如 Qwen2.5-Coder），生成速度体验会有质的飞跃。