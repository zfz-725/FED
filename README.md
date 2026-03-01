# FED: Fast and Efficient Dataset Deduplication with GPU Acceleration

FED is a tool designed for efficient deduplication of data, leveraging modern computational frameworks.

---

## Prerequisites

Before using FED, ensure the following are installed on your system:

- ****CUDA****: For GPU acceleration.
- ****MPI****: For parallel processing.
- ****CMake****: For build automation.

If you use Conda, you can install the prerequisites as follows:

```bash
conda install -c conda-forge mpi4py cmake
conda install -c conda-forge cudatoolkit-dev
```

### System Configuration

The table below summarizes the hardware and software configurations used in this project.
(For more details, please refer to Table 1 in the paper.)

| Component        | Specification                       |
| ---------------- | ----------------------------------- |
| **CPU**          | 1 × AMD EPYC 7502 32-Core Processor |
| **Memory**       | 8 × 64GB DDR4 DIMM                  |
| **GPU**          | 4 × NVIDIA Tesla V100               |
| **OS**           | Ubuntu 20.04.6 (kernel 5.4.0-100)   |
| **Compiler**     | nvcc 12.4, GCC 9.4                  |
| **GPU Driver**   | 520.61.05                           |
| **MPI Version**  | 4.1                                 |
| **CUDA Version** | 12.4                                |


---

## Installation

Follow these steps to build FED:

```bash
mkdir build
cd build
cmake ..
make -j
```

---

## Input, Output, and Execution

### Input and  Hyperparameters

- The input directory should contain multiple `.jsonl` files.
- For large JSONL files, it is recommended to split the data into chunks of size `MAX_LINE` and store them separately before proceeding with the deduplication process.
  - The current code assumes a setting where each JSONL file contains MAX_LINE(=30,000) documents.
  - You can adjust the value of `MAX_LINE` based on the available memory, and update the adjusted value in `src/param.h`.
- This code assumes that the JSONL files contain only the `text` field.  
  - If other fields are present, a parsing process will be required. The parsing logic can be found in `src/util.cpp`, but it is currently commented out in this version.
- Other parameters can be modified in src/param.h.
- The default settings are configured for processing RealNews dataset in our environment.

### Output

- After final duplicate removal, the cleaned dataset is stored in JSONL format within the output directory. 

**Note**: The time taken to save the final output file is excluded from the reported times in the paper. This exclusion ensures a fair comparison with the baseline methods.

### Execution

- ####  Single Process Execution

  To run the tool in single-process mode:

```bash
./main <input_directory> <output_directory>
```

- #### Using MPI for Parallel Execution in Single-node

  To execute our experiment, we utilized **MPI** for parallel processing. Below are the commands for both mpirun and Slurm, which were used to run the code:

Using mpirun

```bash
mpirun -np <num_processes> ./main <input_directory> <output_directory> 
```

Using Slurm

```bash
srun --ntasks=<ntasks>  --cpus-per-task=<cpus-per-task>   --gres=gpu:<num_gpus> --cpu-bind=cores --mpi=pmix --partition=<slurm partition>  ./main <input_directory> <output_directory>
```

**Note**: In our experiment, we observed optimal performance when using mpirun with 4 processes per GPU.
**Note**: While Slurm commands are provided for reference, all experiments in this study were conducted using mpirun.

- #### Multi-node Execution

  To run the tool on multiple nodes with GPU resources:

```bash
mpirun -np <total_num_processes> -hostfile <hostfile> ./main <input_directory> <output_directory> 
```

```bash
srun --ntasks-per-node=<tasks_per_node> --nodes=<num_nodes> —cpus-per-task=<cpus-per-task> --gres=gpu:<num_gpus> --mpi=pmix --partition=<partition_name> ./main <input_directory> <output_directory>
```

**Note**: Our experiment was tested up to 4 nodes. In a multi-node environment, the performance was optimal when the total number of processes was kept under 16. For example, with 4 nodes, we used 4 processes per node.

---

## 两方PSI（私有集合求交）

本项目还包括基于FedRW论文的两方PSI实现，支持用于联邦学习中隐私保护数据重加权的模糊匹配。

### 特性

- **模糊匹配**：使用LSH（局部敏感哈希）查找相似项，不只是精确匹配
- **隐私保护**：两方协议，双方都不需要泄露完整数据集
- **GPU加速**：CUDA加速的签名生成和比较
- **可配置阈值**：可调节的相似度匹配阈值

### 编译PSI

PSI测试程序与主FED程序一起构建：

```bash
mkdir build
cd build
cmake ..
make -j
```

或使用nvcc手动编译：
```bash
nvcc -std=c++17 -O3 -I ../src -I <mpi包含路径> ../src/psi_test.cu ../src/psi.cu ../src/util.cpp ../src/param.cu -o psi_test -L <mpi库路径> -lmpicxx -lmpi
```

### 运行PSI测试

#### 生成测试数据

首先，为双方生成测试数据：

```bash
cd test
python3 generate_test_data.py
```

这将创建：
- `test/party_a.txt` - 参与方A的数据集
- `test/party_b.txt` - 参与方B的数据集

#### 运行PSI协议

执行PSI测试程序：

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<mpi库路径>
./psi_test <参与方a文件> <参与方b文件> <输出目录>
```

示例：
```bash
./psi_test test/party_a.txt test/party_b.txt test/output
```

#### 查看结果

输出目录将包含：

| 文件 | 描述 |
|------|-------------|
| `psi_results.txt` | 人类可读的匹配结果 |
| `a_signatures.bin` | 参与方A的LSH签名（二进制） |
| `b_signatures.bin` | 参与方B的LSH签名（二进制） |
| `buckets.bin` | LSH桶信息（二进制） |
| `candidate_indices.bin` | 匹配的索引对（二进制） |
| `candidates.bin` | 候选对（二进制） |

查看文本结果：
```bash
cat test/output/psi_results.txt
```

使用提供的脚本查看二进制文件：
```bash
python3 test/view_binary.py test/output/a_signatures.bin
python3 test/view_binary.py test/output/buckets.bin
python3 test/view_binary.py test/output/candidate_indices.bin
```

### PSI参数

PSI实现使用以下可配置参数（在 `src/psi.cu` 中）：

- `num_hash`（默认值：128）- LSH的哈希函数数量
- `len_shingle`（默认值：5）- 每个shingle的长度
- `b`（默认值：16）- LSH分桶的波段数
- `threshold`（默认值：0.8）- 匹配的相似度阈值

### 示例输出

```
Party A index: 0, Party B index: 0
Party A text: the quick brown fox jumps over the lazy dog
Similarity threshold: 0.8
---
Party A index: 5, Party B index: 0
Party A text: the quick bgown fox joepssover the lazy dog
Similarity threshold: 0.8
---
Total candidate pairs found: 6
```

这表明PSI协议找到了精确匹配和模糊匹配（具有微小差异的相似字符串）。

### 复杂测试用例

我们还提供了复杂测试用例生成和验证工具：

#### 生成复杂测试数据

```bash
cd test
python3 generate_complex_test.py
```

这将创建包含以下类型的33条测试数据：
- 精确匹配（5条）
- 高相似度模糊匹配（0.9+，5条）
- 中等相似度模糊匹配（0.8+，5条）
- 低相似度模糊匹配（0.7+，5条）
- 长文本（3条，200-300字符）
- 特殊格式文本（5条，包含代码、JSON、错误信息等）
- 完全不同文本（5条）

生成的文件：
- `test/party_a_complex.txt` - 参与方A的复杂测试数据
- `test/party_b_complex.txt` - 参与方B的复杂测试数据

#### 运行复杂测试

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/miniconda3/envs/fed_env/lib
./psi_test test/party_a_complex.txt test/party_b_complex.txt test/output_complex
```

#### 验证测试结果

使用验证脚本分析匹配结果：

```bash
python3 test/verify_psi_results.py test/output_complex/a_signatures.bin test/output_complex/b_signatures.bin test/output_complex/candidate_indices.bin
```

验证脚本将输出：
- 每个候选对的详细匹配信息
- Jaccard相似度计算
- 匹配类型分类（精确匹配、高/中/低相似度）
- 统计摘要和准确率

#### 示例验证输出

```
================================================================================
统计摘要
================================================================================
总候选对数: 11
精确匹配 (相似度=1.0): 6
高相似度 (0.8-1.0): 0
中等相似度 (0.6-0.8): 4
低相似度 (<0.6): 1

准确率 (相似度>=0.6): 90.91%
```

### 中文测试用例

我们还提供了中文测试数据生成和验证工具，用于测试PSI在中文文本上的模糊匹配效果：

#### 生成中文测试数据

```bash
cd test
python3 generate_chinese_test.py
```

这将创建包含以下类型的33条中文测试数据：
- 精确匹配（5条）
- 高相似度模糊匹配（0.9+，5条）
- 中等相似度模糊匹配（0.8+，5条）
- 低相似度模糊匹配（0.7+，5条）
- 长文本（3条，200-300字符）
- 特殊格式文本（5条，包含代码、JSON、错误信息等）
- 完全不同文本（5条）

生成的文件：
- `test/party_a_chinese.txt` - 参与方A的中文测试数据
- `test/party_b_chinese.txt` - 参与方B的中文测试数据

#### 运行中文测试

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/miniconda3/envs/fed_env/lib
./psi_test test/party_a_chinese.txt test/party_b_chinese.txt test/output_chinese
```

#### 验证中文测试结果

使用验证脚本分析中文测试的匹配结果：

```bash
python3 test/verify_psi_results.py test/output_chinese/a_signatures.bin test/output_chinese/b_signatures.bin test/output_chinese/candidate_indices.bin
```

#### 示例中文测试输出

```
================================================================================
PSI测试结果验证
================================================================================

1. 读取参与方A的数据...
   读取了 33 条记录

2. 读取参与方B的数据...
   读取了 33 条记录

3. 读取候选对...
   找到 11 个候选对

================================================================================
匹配结果分析
================================================================================

候选对 1:
  参与方A索引: 0
  参与方B索引: 0
  相似度: 1.0000
  参与方A文本: 机器学习是人工智能的核心技术
  参与方B文本: 机器学习是人工智能的核心技术
  [精确匹配]

候选对 2:
  参与方A索引: 24
  参与方B索引: 24
  相似度: 0.9219
  参与方A文本: 用户ID：12345 | 时间戳：2024-01-15 14:30:00 | 操作：登录
  参与方B文本: 时间戳：2024-01-15 14:30:00 | 用户ID：12345 | 操作：登录成功
  [高相似度]

================================================================================
统计摘要
================================================================================
总候选对数: 11
精确匹配 (相似度=1.0): 9
高相似度 (0.8-1.0): 1
中等相似度 (0.6-0.8): 1
低相似度 (<0.6): 0

准确率 (相似度>=0.6): 100.00%
```

### 故障排除

#### 错误：error while loading shared libraries: libmpi.so.12

**原因**：找不到MPI共享库

**解决方案**：设置LD_LIBRARY_PATH环境变量

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/miniconda3/envs/fed_env/lib
```

或者添加到~/.bashrc中使其永久生效：

```bash
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/miniconda3/envs/fed_env/lib' >> ~/.bashrc
source ~/.bashrc
```
