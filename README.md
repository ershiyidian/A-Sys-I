# A-Sys-I Project: A Telescope for AI Model Internals
# A-Sys-I 项目：AI 模型内部动态观测望远镜

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Status](https://img.shields.io/badge/Status-Phase%201%20Infrastructure-orange)

---
## 1. Introduction / 项目简介

**(EN)**
**Vision:** To build a precise, reliable, and non-intrusive "telescope" (A-Sys-I) for observing the internal dynamics (e.g., activations, gradients) of AI models during runtime.
This repository provides the core infrastructure code for the A-Sys-I project. The goal of Phase 1 is to deliver a robust, SLA-compliant, and backward-compatible infrastructure library, validated through rigorous benchmarks. The system is designed with a clear separation between the model being observed (the subject) and the observation system itself (A-Sys-I), operating in two distinct modes: High-Performance Computing (HPC) for maximum throughput and predictable latency, and CONSUMER mode for broader compatibility and ease of use with graceful performance degradation. The primary use-case demonstrated is capturing layer activations from a host model and training Sparse Autoencoders (SAEs).

**(CN)**
**项目愿景:** 建造一台精密、可靠、非侵入式的“望远镜”(A-Sys-I)，用于观测 AI 模型运行时的内部动态（如激活值、梯度等）。
本代码库提供了 A-Sys-I 项目的核心基础设施代码。第一阶段的目标是交付一个经过严格基准测试验证、满足服务等级协议（SLA）、健壮且向下兼容的基础设施库。系统设计的核心原则是“观测-主体分离”，即被观测的模型（主体）与观测系统本身（A-Sys-I）严格解耦。系统支持两种运行模式：高性能计算（HPC）模式，追求最大吞吐量和可预测的延迟；以及 CONSUMER 模式，牺牲部分性能以保障更广泛的兼容性和易用性（优雅降级）。项目展示的核心用例是从宿主模型中捕获各层激活值，并用于训练稀疏自编码器（SAEs）。

---
## Table of Contents / 目录
1.  [Introduction / 项目简介](#1-introduction--项目简介)
2.  [Core Philosophy / 核心哲学](#2-core-philosophy--核心哲学)
3.  [Key Features / 关键特性](#3-key-features--关键特性)
4.  [Architecture Overview / 架构总览](#4-architecture-overview--架构总览)
    *   [Architecture Diagram / 架构图](#architecture-diagram--架构图)
    *   [Data Flow / 核心数据流](#data-flow--核心数据流)
    *   [Control Flow / 核心控制流](#control-flow--核心控制流)
5.  [Operating Modes: HPC vs. CONSUMER / 运行模式: HPC vs. CONSUMER](#5-operating-modes-hpc-vs-consumer--运行模式-hpc-vs-consumer)
6.  [Installation / 安装指南](#6-installation--安装指南)
     * [Prerequisites / 前提条件](#prerequisites--前提条件)
     * [Pip Installation / Pip 安装](#pip-installation--pip-安装)
     * [Docker Installation / Docker 安装](#docker-installation--docker-安装)
7.  [Quick Start / 快速开始](#7-quick-start--快速开始)
8.  [Configuration / 配置说明](#8-configuration--配置说明)
9.  [Benchmarking & SLA / 基准测试与服务等级协议](#9-benchmarking--sla--基准测试与服务等级协议)
10. [Observability & Fault Tolerance / 可观测性与容错性](#10-observability--fault-tolerance--可观测性与容错性)
11. [Project Structure / 项目结构](#11-project-structure--项目结构)
12. [Contributing / 贡献指南](#12-contributing--贡献指南)
13. [License / 许可证](#13-license--许可证)

---
## 2. Core Philosophy / 核心哲学

**(EN)**
The design of A-Sys-I is guided by the following principles:
1.  **Separation (Observer-Subject):** The highest principle. The observation system must minimally interfere with the host model's execution.
2.  **Predictability:** `HPC` mode must meet stringent Service Level Agreements (SLAs) for latency and throughput.
3.  **Graceful Degradation:** `CONSUMER` mode guarantees usability and availability on diverse hardware, trading off performance.
4.  **Config-Driven:** All behaviours, parameters, and mode-switching are controlled via configuration files (`.yaml`); zero code modification is required to run different experiments.
5.  **Observability-First:** The A-Sys-I system itself must be transparent, exposing key metrics, logs, and health status.
6.  **Design for Failure:** Assume any component can fail; implement monitoring (Watchdog), heartbeats, and restart mechanisms.

**(CN)**
A-Sys-I 的设计遵循以下核心原则：
1.  **观测-主体分离 (Separation):** 工程的最高原则。观测系统对宿主模型运行的干扰必须降到最低。
2.  **性能可预测 (Predictability):** `HPC` 模式必须满足关于延迟和吞吐量的严苛服务等级协议（SLA）。
3.  **优雅降级 (Graceful Degradation):** `CONSUMER` 模式在通用硬件上保证系统可用性，允许牺牲部分性能。
4.  **配置驱动 (Config-Driven):** 所有行为、参数和模式切换均由配置文件(`.yaml`)控制，运行不同实验无需修改任何代码。
5.  **可观测性优先 (Observability-First):** A-Sys-I 系统自身必须是透明的，能向上暴露关键指标、日志和健康状态。
6.  **为失败设计 (Design for Failure):** 假设任何组件都可能崩溃，系统内置监控（看门狗Watchdog）、心跳检测和自动重启机制。

---
## 3. Key Features / 关键特性

**(EN)**
*   **Dual-Mode Operation:** Seamless switching between `HPC` (optimised C++/CUDA core) and `CONSUMER` (pure-Python) modes via configuration.
*   **Non-Intrusive Hooking:** Leverages `torch.nn.Module.register_forward_hook` to capture activations with minimal overhead. Backpressure mechanism dynamically adjusts sampling rate.
*    **High-Performance Data Bus:** 
    * `HPC`: Lock-free C++ Single-Producer-Multiple-Consumer (SPMC) sharded ring buffer (`CppShardedSPMCBus`) using shared memory for zero-copy tensor transfer, avoiding Python GIL and serialization overhead.
    * `CONSUMER`: Python `multiprocessing.Queue` based implementation (`PythonQueueBus`) for maximum compatibility.
*   **Asynchronous Processing:** Activation capture, pre-processing (quantization/compression via GPU kernels in HPC mode), SAE training, and data archiving run in separate processes/streams, decoupled from the host model's critical path.
*   **Fault Tolerance:** A `Watchdog` process monitors component heartbeats and automatically restarts failed workers (e.g., `SAETrainerWorker`).
*   **Rich Observability:**
     * `HPC`: Prometheus endpoint for metric scraping, integrated with Grafana/Loki via `docker-compose`.
     * `CONSUMER`: Logging, CSV, and TensorBoard output.
*   **Resource Management (HPC):** CPU core pinning and NUMA node memory binding for `SAETrainerWorker` processes to ensure predictable performance and minimize resource contention via `resource_manager`.
*   **Configuration Management:** Hierarchical YAML configuration with strict validation using `Pydantic`.
*   **Reproducibility:** Docker environment ensures consistent setup across different machines.

**(CN)**
*   **双模式运行:** 通过配置即可在 `HPC`（C++/CUDA核心优化）和 `CONSUMER`（纯Python）模式间无缝切换。
*   **非侵入式钩子:** 利用 `torch.nn.Module.register_forward_hook` 捕获激活值，开销极低。内置背压（Backpressure）机制可动态调整采样率。
*   **高性能数据总线 (Data Bus):**
    * `HPC`: 使用基于共享内存的无锁 C++ 单生产者多消费者（SPMC）分片环形缓冲区 (`CppShardedSPMCBus`)，实现张量零拷贝传输，避免 Python GIL 和序列化开销。
    * `CONSUMER`: 使用基于 Python `multiprocessing.Queue` 的实现 (`PythonQueueBus`)，提供最大兼容性。
*   **全异步处理:** 激活捕获、预处理（HPC模式下使用GPU Kernels量化/压缩）、SAE 训练和数据归档均在独立进程/CUDA流中运行，与宿主模型的关键路径解耦。
*   **容错能力:** `Watchdog`（看门狗）进程监控各组件心跳，并能自动重启失败的工作进程（如 `SAETrainerWorker`）。
*   **丰富的可观测性:**
     * `HPC`: 提供 Prometheus 指标端点，可通过 `docker-compose` 与 Grafana/Loki 集成。
     * `CONSUMER`: 提供日志、CSV 文件和 TensorBoard 输出。
*   **资源管理 (HPC):** 通过 `resource_manager` 对 `SAETrainerWorker` 进程进行 CPU 核心绑定和 NUMA 节点内存绑定，确保性能可预测性并最小化资源争用。
*   **配置管理:** 使用分层 YAML 配置文件，并通过 `Pydantic` 进行严格的类型和结构验证。
*    **可复现性:** 提供 Docker 环境，确保在不同机器上环境配置的一致性。

---
## 4. Architecture Overview / 架构总览

**(EN)**
The system follows a modular, decoupled architecture. The `ExperimentPipeline` acts as the central orchestrator, instantiating and managing the lifecycle of all components based on the `MasterConfig`. Components interact via well-defined interfaces (`BaseDataBus`, `BaseMonitor`) and data structures (`ActivationPacket`), with concrete implementations selected by Factory patterns based on the `run_profile` (HPC/CONSUMER).

**(CN)**
系统采用模块化、解耦的架构。`ExperimentPipeline` 作为中央编排器，根据 `MasterConfig` 实例化并管理所有组件的生命周期。各组件通过定义清晰的接口（`BaseDataBus`, `BaseMonitor`）和数据结构（`ActivationPacket`）进行交互，具体实现类由工厂模式根据 `run_profile` (HPC/CONSUMER) 动态选择。

### Architecture Diagram / 架构图
```mermaid
graph TD
    %% --- 配置文件 ---
     subgraph 配置 (configs/)
        CFG_HPC[profile_hpc.yaml]
        CFG_CONS[profile_consumer.yaml]
        CFG_BASE[base.yaml]
     end

    %% --- 入口 ---
    subgraph 入口 (scripts/)
       ENTRY[run_experiment.py] -- 加载 --> CFG_HPC
       ENTRY -- 加载 --> CFG_CONS
       ENTRY -- 加载 --> CFG_BASE
       ENTRY -- 实例化 & 运行 --> PL
    end

     %% --- 核心编排 ---
    subgraph 编排 (src/asys_i/orchestration/)
        PL[pipeline.py<br>ExperimentPipeline] -- 验证配置 --> CL
        PL -- 创建(工厂) --> DB_F
        PL -- 创建(工厂) --> MON_F
        PL -- 创建 --> HOST
        PL -- 创建 --> HOOK
        PL -- 创建 --> TRAIN_MGR
        PL -- 创建 --> WD
        PL -- 创建 --> ARCH
    end
      CL[config_loader.py<br>MasterConfig]

    %% --- 数据流组件 ---
    subgraph 核心组件 (src/asys_i/components/)
        HOST[ppo_host.py<br>PPOHostProcess]
        HOOK[activation_hooker.py<br>ActivationHooker]
        DB_INT[data_bus_interface.py<br>BaseDataBus]
        DB_F[data_bus_factory.py]-- profile? --> DB_HPC & DB_CONS
        DB_HPC[data_bus_hpc.py<br>CppShardedSPMCBus] -- 使用 --> CPP
        DB_CONS[data_bus_consumer.py<br>PythonQueueBus]
        SAE_MOD[sae_model.py<br>SparseAutoencoder]
        TRAIN_MGR[sae_trainer.py<br>SAETrainerManager] 
        TRAIN_W[sae_trainer.py<br>SAETrainerWorker] -- 使用 --> SAE_MOD
        ARCH[archiver.py<br>ArchiverWorker]

         HOST -- 提供模型给 --> HOOK
         HOST -- 运行循环 --> HOST
         HOOK -- attach到 --> HOST
    end
     DB_F -- 实现 --> DB_INT
     MON_F -- 实现 --> MON_INT

    %% --- 数据包定义 ---
     subgraph 通用 (src/asys_i/common/)
       PACK[types.py<br>ActivationPacket]
     end
     HOOK -- 产生 --> PACK
     DB_INT -- 定义接口传输 --> PACK
     TRAIN_W -- 消费 --> PACK
     ARCH -- 消费 --> PACK

     %% --- HPC 优化 ---
      subgraph HPC (src/asys_i/hpc/)
        CPP[cpp_ringbuffer/*<br>C++ Core & Bindings]
        GPU[gpu_kernels.py<br>Quant/Compress]
        RES[resource_manager.py<br>CPU/NUMA Binding]
        HOOK -- (HPC) 使用 --> GPU
        TRAIN_MGR -- (HPC) 使用 --> RES
      end

    %% --- 监控 ---
     subgraph 监控 (src/asys_i/monitoring/)
        MON_INT[monitor_interface.py<br>BaseMonitor]
        MON_F[monitor_factory.py] -- profile? --> MON_HPC & MON_CONS
        MON_HPC[monitor_hpc.py<br>PrometheusMonitor]
        MON_CONS[monitor_consumer.py<br>LoggingCSVMonitor]
        WD[watchdog.py<br>WatchdogProcess]

        ALL_MOD{{所有模块}} -- 上报指标/心跳 --> MON_INT
        WD -- 检查心跳/重启 --> TRAIN_W & ARCH
     end
    
    %% --- 基础设施 ---
    subgraph 基础设施
         DOCK[docker/*]
         TESTS[tests/*]
         PYPROJ[pyproject.toml<br>Dependencies, Build C++]
    end

%% --- 数据流向 ---
   HOOK -- 1. 捕获/预处理/推送 PACK --> DB_INT
   TRAIN_W -- 2. 拉取 PACK --> DB_INT
   ARCH -- 2. 拉取 PACK --> DB_INT
   TRAIN_W --> SAE_MOD
   
%% --- 控制流向 ---
    DB_INT -- 背压信号 --> HOOK
    PL -- setup/run/shutdown --> ALL_MOD

    style PL fill:#ccf,stroke:#333,stroke-width:2px
    style DB_INT fill:#f9f,stroke:#333,stroke-width:1px
     style MON_INT fill:#ff9,stroke:#333,stroke-width:1px
```
### Data Flow / 核心数据流
**(EN)**
1.  `PPOHostProcess` runs the host model's forward pass.
2.  `ActivationHooker` hooks are triggered.
    *   `HPC`: On a low-priority `cuda.Stream`, uses `gpu_kernels` (quantise/compress), `cudaMemcpyAsync` to pinned memory, creates `ActivationPacket` (with tensor reference), and pushes to `CppShardedSPMCBus` (C++ ring buffer via shared memory).
    *   `CONSUMER`: Calls `tensor.detach().cpu()`, creates `ActivationPacket` (containing the tensor), and pushes to `PythonQueueBus` (`multiprocessing.Queue`).
3.  `SAETrainerWorker` (separate process) polls `DataBus.pull_batch` to get `ActivationPacket`s and performs a training `step` on the `SparseAutoencoder`.
4.  `ArchiverWorker` (separate process) polls `DataBus.pull_batch` and asynchronously persists `ActivationPacket`s to storage.
5.  All components periodically report metrics and heartbeats via the `BaseMonitor` interface.
6.  If `DataBus.push` fails (buffer full), a backpressure signal causes `ActivationHooker` to dynamically reduce its sampling rate.

**(CN)**
1.  `PPOHostProcess` 运行宿主模型的前向传播。
2.  `ActivationHooker` 的钩子函数被触发。
    *   `HPC`: 在低优先级 `cuda.Stream` 上，调用 `gpu_kernels` (量化/压缩)，通过 `cudaMemcpyAsync` 拷贝到锁页内存(Pinned Memory)，生成含张量引用的 `ActivationPacket`，并通过 `CppShardedSPMCBus` 推送到基于共享内存的 C++ 环形缓冲区。
    *   `CONSUMER`: 调用 `tensor.detach().cpu()`，生成包含张量数据的 `ActivationPacket`，并通过 `PythonQueueBus` 推送到 `multiprocessing.Queue`。
3.  `SAETrainerWorker`（独立进程）轮询 `DataBus.pull_batch` 获取 `ActivationPacket` 批次，并在 `SparseAutoencoder` 模型上执行训练 `step`。
4.  `ArchiverWorker`（独立进程）轮询 `DataBus.pull_batch` 获取 `ActivationPacket`，并异步地将其持久化到存储。
5.  所有组件定期通过 `BaseMonitor` 接口上报指标和心跳。
6.  如果 `DataBus.push` 失败（缓冲区满），将产生背压信号，`ActivationHooker` 捕获该信号并动态降低采样率。

### Control Flow / 核心控制流
**(EN)**
1. `scripts/run_experiment.py` is the entry point.
2. It loads and validates `MasterConfig` via `config_loader.py`.
3. `ExperimentPipeline` is instantiated.
4. `pipeline.setup()`: Uses factories (`create_monitor`, `create_data_bus`) to instantiate correct Monitor and DataBus based on config, creates Host, Hooker, TrainerManager, Archiver, Watchdog. Hooks are attached. Workers are initialised (CPU/NUMA binding in HPC).
5. `pipeline.run()`: Starts Watchdog, TrainerManager, Archiver, and finally runs the blocking `PPOHostProcess.run_training_loop()`.
6. `pipeline.shutdown()`: Called on completion or interruption (`SIGINT`/`SIGTERM` handling), ensures orderly shutdown: stop signals, worker joins, data bus release, monitor flushing.
7. `Watchdog` monitors heartbeats collected by `Monitor` and calls `SAETrainerManager.restart_worker` on timeout.

**(CN)**
1. `scripts/run_experiment.py` 是程序主入口。
2. 它通过 `config_loader.py` 加载并验证 `MasterConfig` 配置对象。
3. 实例化 `ExperimentPipeline` 编排器。
4. `pipeline.setup()`: 调用工厂函数（`create_monitor`, `create_data_bus`），根据配置创建正确的 Monitor 和 DataBus 实例，并创建 Host, Hooker, TrainerManager, Archiver, Watchdog。 附加钩子，初始化 Workers（HPC模式下进行CPU/NUMA绑定）。
5. `pipeline.run()`: 启动 Watchdog, TrainerManager, Archiver，最后运行阻塞的 `PPOHostProcess.run_training_loop()`。
6. `pipeline.shutdown()`: 在程序完成或被中断（处理 `SIGINT`/`SIGTERM` 信号）时调用，确保有序关闭：发送停止信号、等待worker join、释放数据总线资源、刷新监控数据。
7. `Watchdog` 监控 `Monitor` 收集的心跳数据，并在超时时调用 `SAETrainerManager.restart_worker` 重启对应工作进程。

---
## 5. Operating Modes: HPC vs. CONSUMER / 运行模式: HPC vs. CONSUMER

**(EN)**
The system behaviour is dramatically different based on `config.run_profile`.
| Feature | HPC Mode | CONSUMER Mode |
| :--- | :--- | :--- |
| **Goal** | Predictable Performance, Max Throughput, SLA | Availability, Compatibility, Ease of Use |
| **Philosophy** | Predictability | Graceful Degradation |
| **Data Bus** | `CppShardedSPMCBus`: C++ Lock-free, Shared Memory, Zero-Copy | `PythonQueueBus`: `multiprocessing.Queue`, Serialization overhead |
| **Hook Process**| Async GPU Kernel (Quant/Compress), `cudaMemcpyAsync` to PinnedMem| Sync `tensor.detach().cpu()` copy |
| **Data Transfer**| Tensor Reference + Shared Memory (No GIL) | Tensor Object Serialization/Deserialization (GIL involved)|
| **Monitoring** | `PrometheusMonitor`: Pull-based metrics endpoint | `LoggingCSVMonitor`: Log file, CSV, Tensorboard |
| **Resource Mgmt**| CPU Core Pinning, NUMA memory binding (`psutil`, `numactl`)| OS Default Scheduling |
| **Dependencies** | C++ compiler, `pybind11`, `cmake`, `prometheus-client`, CUDA-devel| Pure Python dependencies |
| **Fault Tolerance**| Watchdog + Worker Restart | Watchdog + Worker Restart |
| **Orchestration**| `docker-compose` (App + Prometheus + Grafana) | Script execution |

**(CN)**
系统行为根据 `config.run_profile` 的设置有显著区别。
| 特性 | HPC 模式 | CONSUMER 模式 |
| :--- | :--- | :--- |
| **目标** | 性能可预测, 最大吞吐量, 满足SLA | 可用性, 兼容性, 易用性 |
| **遵循哲学** | Predictability (性能可预测) | Graceful Degradation (优雅降级)|
| **数据总线** | `CppShardedSPMCBus`: C++无锁, 共享内存, 零拷贝 | `PythonQueueBus`: `multiprocessing.Queue`, 存在序列化开销 |
| **钩子处理** | 异步GPU Kernel(量化/压缩), `cudaMemcpyAsync`到锁页内存 | 同步 `tensor.detach().cpu()` 拷贝 |
| **数据传输** | 张量引用+共享内存 (绕过GIL) | 张量对象序列化/反序列化 (受GIL影响)|
| **监控方式** | `PrometheusMonitor`: 被动拉取(Pull)指标端点 | `LoggingCSVMonitor`: 日志文件, CSV, Tensorboard |
| **资源管理** | CPU核心绑定, NUMA内存绑定 (`psutil`, `numactl`) | 操作系统默认调度 |
| **依赖项** | C++编译器, `pybind11`, `cmake`, `prometheus-client`, CUDA开发库| 纯Python依赖 |
| **容错机制** | Watchdog + Worker 重启 | Watchdog + Worker 重启 |
| **推荐编排**| `docker-compose` (应用+Prometheus+Grafana) | 直接脚本运行 |

---
## 6. Installation / 安装指南

### Prerequisites / 前提条件
*   Python 3.10+
*   Git
*   Conda / Mamba (Recommended)
*   **HPC Mode Only:**
    *   NVIDIA GPU with CUDA Toolkit (e.g., 11.8+) installed (`nvcc` must be in PATH).
    *   C++ build toolchain (`build-essential`, `cmake`, `ninja`).
    *   Docker & Docker Compose (Recommended for monitoring stack).
    *   Linux OS (for CPU/NUMA binding).

### Pip Installation / Pip 安装

**(EN)**
Managed via `pyproject.toml`.
1. Clone the repository:
   ```bash
    git clone <repository-url>
    cd A-Sys-I
   ```
2. Create and activate conda environment:
    ```bash
     conda create -n asys-i python=3.10 -y
     conda activate asys-i
    ```
3. Install project:
    * **For CONSUMER mode / Development:**
       ```bash
        pip install -e .[dev] 
       ```
       (`-e` for editable mode, `[dev]` includes linting/testing tools).
    * **For HPC mode:**
       ```bash
       # Ensure nvcc and C++ compiler are available!
       pip install -e .[hpc,dev]
       ```
        This command automatically triggers the build of the C++ extension (`cpp_ringbuffer`) via `pybind11` as defined in `pyproject.toml`. The `[hpc]` extra installs HPC-specific dependencies like `prometheus-client`.

**(CN)**
依赖通过 `pyproject.toml` 管理。
1. 克隆代码库:
   ```bash
    git clone <repository-url>
    cd A-Sys-I
   ```
2. 创建并激活 conda 环境:
    ```bash
     conda create -n asys-i python=3.10 -y
     conda activate asys-i
    ```
3. 安装项目:
    * **CONSUMER 模式 / 开发环境:**
       ```bash
        pip install -e .[dev] 
       ```
       (`-e` 表示可编辑模式, `[dev]` 包含代码检查/测试工具)。
    * **HPC 模式:**
       ```bash
       # 请确保 nvcc 和 C++ 编译器已安装且可用！
       pip install -e .[hpc,dev]
       ```
        此命令将根据 `pyproject.toml` 中的定义，通过 `pybind11` 自动触发 C++ 扩展 (`cpp_ringbuffer`) 的编译构建。`[hpc]` 选项会安装 HPC 模式专用的依赖（如 `prometheus-client`）。

### Docker Installation / Docker 安装

**(EN)**
Provides a reproducible environment, especially for HPC mode.
* Build image:
   ```bash
    # Build for CONSUMER mode
    docker build -t asys-i:consumer --build-arg profile=CONSUMER -f docker/Dockerfile .
    # Build for HPC mode (requires nvidia-docker runtime)
    docker build -t asys-i:hpc --build-arg profile=HPC -f docker/Dockerfile .
   ```
* Run (HPC example with monitoring stack):
  Modify `docker/docker-compose.yml` to mount your configs/output paths, then:
   ```bash
    docker-compose -f docker/docker-compose.yml up
   ```
  This starts the app, Prometheus, and Grafana. Access Grafana at `http://localhost:3000` and Prometheus at `http://localhost:9090`.

**(CN)**
提供可复现的环境，特别推荐用于 HPC 模式。
* 构建镜像:
   ```bash
    # 构建 CONSUMER 模式镜像
    docker build -t asys-i:consumer --build-arg profile=CONSUMER -f docker/Dockerfile .
    # 构建 HPC 模式镜像 (需要 nvidia-docker runtime)
    docker build -t asys-i:hpc --build-arg profile=HPC -f docker/Dockerfile .
   ```
* 运行 (HPC 模式 + 监控栈示例):
  修改 `docker/docker-compose.yml` 挂载您的配置和输出目录，然后运行:
   ```bash
    docker-compose -f docker/docker-compose.yml up
   ```
  这将启动应用、Prometheus 和 Grafana。访问 Grafana: `http://localhost:3000`；访问 Prometheus: `http://localhost:9090`。

---
## 7. Quick Start / 快速开始

**(EN)**
After installation, the `pip install` process creates CLI entry points defined in `pyproject.toml` (`asys-i-run`, `asys-i-bench`).
*   Run with CONSUMER profile:
    ```bash
    # Activate conda env if not using docker
    asys-i-run --config configs/profile_consumer.yaml
    ```
*   Run with HPC profile:
     ```bash
    # Activate conda env if not using docker
    asys-i-run --config configs/profile_hpc.yaml
    ```
   Metrics/Logs/Checkpoints will be saved according to the configuration.

**(CN)**
安装完成后， `pip install` 过程会根据 `pyproject.toml` 的定义创建命令行入口 (`asys-i-run`, `asys-i-bench`)。
*   以 CONSUMER 模式运行:
    ```bash
    # 如果不是用 docker，请先激活 conda 环境
    asys-i-run --config configs/profile_consumer.yaml
    ```
*   以 HPC 模式运行:
     ```bash
    # 如果不是用 docker，请先激活 conda 环境
     asys-i-run --config configs/profile_hpc.yaml
    ```
   指标、日志、模型检查点将根据配置文件中的路径保存。
---
## 8. Configuration / 配置说明

**(EN)**
The system is entirely Config-Driven. 
- `configs/base.yaml`: Contains default settings.
- `configs/profile_consumer.yaml`: Overrides base for CONSUMER mode (`run_profile: CONSUMER`, `monitor_type: csv`, `data_bus: python_queue`).
- `configs/profile_hpc.yaml`: Overrides base for HPC mode (`run_profile: HPC`, `monitor_type: prometheus`, `data_bus: cpp_spmc`, CPU/NUMA maps).
`src/asys_i/config_loader.py` uses `Pydantic` to define the `MasterConfig` schema, load, merge (specific profile overrides base), and validate the YAML configuration, ensuring type safety and structural integrity before `ExperimentPipeline` initialisation. The `run_profile` field is the key determinant for factory patterns.

**(CN)**
系统完全采用配置驱动。
- `configs/base.yaml`: 包含所有通用默认设置。
- `configs/profile_consumer.yaml`: 覆盖 base 配置，设定 CONSUMER 模式参数 (`run_profile: CONSUMER`, `monitor_type: csv`, `data_bus: python_queue`)。
- `configs/profile_hpc.yaml`: 覆盖 base 配置，设定 HPC 模式参数 (`run_profile: HPC`, `monitor_type: prometheus`, `data_bus: cpp_spmc`, CPU/NUMA 映射表)。
`src/asys_i/config_loader.py` 使用 `Pydantic` 库定义了 `MasterConfig` 的数据结构模型(schema)，负责加载、合并（特定 profile 覆盖 base）和验证 YAML 配置，在 `ExperimentPipeline` 初始化之前确保配置的类型安全和结构完整性。`run_profile` 字段是工厂模式选择具体实现类的关键依据。

---
## 9. Benchmarking & SLA / 基准测试与服务等级协议

**(EN)**
A key objective of Phase 1 is SLA validation. `src/asys_i/orchestration/benchmark_suite.py` defines tests to verify the infrastructure's performance characteristics, focusing on:
*   **Latency:** P99 latency of the hook and data push path (`hook_latency_ms`).
*   **Throughput:** Max data transfer rate (GB/s or Packets/s) through the `DataBus` without packet drops.
*   **Isolation (HPC):** Measuring the performance impact on a baseline task when A-Sys-I components (bound to different cores) are active, validating resource manager effectiveness.
 
Run benchmarks using:
```bash
 asys-i-bench --config configs/benchmark.yaml
```
 The suite generates a report comparing results against predefined SLA targets.

**(CN)**
第一阶段的关键目标是 SLA 验证。`src/asys_i/orchestration/benchmark_suite.py` 定义了一系列测试用例，以验证基础设施的性能特征，重点关注：
*   **延迟 (Latency):** 钩子函数执行及数据推送路径的 P99 延迟 (`hook_latency_ms`)。
*   **吞吐量 (Throughput):** `DataBus` 在不丢包情况下的最大数据传输速率 (GB/s 或 Packets/s)。
*   **隔离性 (Isolation) (HPC):** 当 A-Sys-I 组件（绑定到不同核心）运行时，测量其对基线任务性能的影响，以验证资源管理器的有效性。

运行基准测试：
```bash
 asys-i-bench --config configs/benchmark.yaml
```
测试套件将生成一份报告，将测试结果与预设的 SLA 目标进行对比。

---
## 10. Observability & Fault Tolerance / 可观测性与容错性

**(EN)**
Adhering to "Observability-First" and "Design for Failure":
- **Metrics/Heartbeats**: All components report metrics (e.g., `data_bus_push_count`, `data_bus_drop_count`, `hook_latency_ms`, `sae_loss`) and regular heartbeats through the `BaseMonitor` interface.
- **Monitoring**: 
  - `HPC`: `PrometheusMonitor` exposes a `/metrics` endpoint.
  - `CONSUMER`: `LoggingCSVMonitor` writes to logs, CSV, and TensorBoard.
- **Watchdog**: `src/asys_i/monitoring/watchdog.py` runs in a separate thread/process, periodically checking the last heartbeat timestamp for each critical component (`SAETrainerWorker`, `ArchiverWorker`) via `monitor.get_heartbeats()`. If a component times out, the Watchdog triggers its restart logic (e.g., `trainer_manager.restart_worker(component_id)`), ensuring system resilience.

**(CN)**
秉承“可观测性优先”和“为失败设计”原则：
- **指标/心跳**: 所有组件均通过 `BaseMonitor` 接口上报关键指标（如 `data_bus_push_count`, `data_bus_drop_count`, `hook_latency_ms`, `sae_loss`）和周期性心跳。
- **监控实现**:
  - `HPC`: `PrometheusMonitor` 暴露 `/metrics` 端点供 Prometheus 抓取。
  - `CONSUMER`: `LoggingCSVMonitor` 将数据写入日志、CSV 文件和 TensorBoard。
- **看门狗 (Watchdog)**: `src/asys_i/monitoring/watchdog.py` 在独立线程/进程中运行，通过 `monitor.get_heartbeats()` 定期检查关键组件（`SAETrainerWorker`, `ArchiverWorker`）的最近心跳时间戳。如果组件心跳超时，Watchdog 将触发其重启逻辑（如 `trainer_manager.restart_worker(component_id)`），保障系统的自愈能力和弹性。

---
## 11. Project Structure / 项目结构
```
A-Sys-I/
├── configs/                 # Configuration files (base.yaml, profile_*.yaml)
├── docker/                  # Dockerfile, docker-compose.yml, entrypoint.sh
├── scripts/                 # Entry points (run_experiment.py, run_benchmark.py)
├── src/asys_i/              # Main source package
│   ├── common/              # Shared types (ActivationPacket)
│   ├── components/          # Core functional modules (DataBus, Hooker, SAE, Trainer, Archiver, Host, Factories)
│   ├── hpc/                 # HPC-specific optimisations
│   │   ├── cpp_ringbuffer/  # C++ SPMC queue core and pybind11 bindings
│   │   ├── gpu_kernels.py   # CUDA pre-processing
│   │   └── resource_manager.py # CPU/NUMA binding
│   ├── monitoring/          # Monitoring and Watchdog (Interface, Prometheus, CSV, Factory, Watchdog)
│   ├── orchestration/       # System coordination (ExperimentPipeline, BenchmarkSuite)
│   ├── utils.py             # Helper functions
│   └── config_loader.py     # Pydantic config models and loading logic
├── tests/                   # Pytest unit and integration tests
├── pyproject.toml           # Project metadata, dependencies, build config, CLI scripts
└── README.md                # This file
```
---
## 12. Contributing / 贡献指南
**(EN)**
Contributions are welcome! Please follow the standard GitHub flow: fork, create a feature branch, commit, pass tests (`pytest`), ensure code style (`black`, `isort`, `flake8`, `mypy`), and submit a Pull Request.
**(CN)**
欢迎贡献代码！请遵循标准 GitHub 流程：Fork 项目，创建特性分支，提交代码，确保通过测试 (`pytest`) 并符合代码风格 (`black`, `isort`, `flake8`, `mypy`)，然后提交 Pull Request。

---
 ## 13. License / 许可证
**(EN)**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
**(CN)**
本项目采用 MIT 许可证 - 详情请见 [LICENSE](LICENSE) 文件。
