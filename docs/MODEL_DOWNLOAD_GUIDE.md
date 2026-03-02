# 模型下载指南

## 模型来源

所有模型都来自 **HuggingFace Hub**：https://huggingface.co/

## 项目所需模型

### 小型模型（必需，约 900 MB）

1. **gpt2** (~500 MB)
   - 链接：https://huggingface.co/gpt2
   - 用途：测试脚本、示例代码
   - 必需度：⭐⭐⭐

2. **uer/gpt2-chinese-cluecorpussmall** (~400 MB)
   - 链接：https://huggingface.co/uer/gpt2-chinese-cluecorpussmall
   - 用途：多任务学习、指令微调、模型比较
   - 必需度：⭐⭐⭐

### 中型模型（可选，约 810 MB）

3. **uer/gpt2-chinese-poem** (~450 MB)
   - 链接：https://huggingface.co/uer/gpt2-chinese-poem
   - 用途：模型比较
   - 必需度：⭐⭐

4. **facebook/wav2vec2-base** (~360 MB)
   - 链接：https://huggingface.co/facebook/wav2vec2-base
   - 用途：语音识别、口音分类
   - 必需度：⭐⭐

### 大型模型（按需，约 14 GB）

5. **Qwen/Qwen-7B-Chat** (~14 GB)
   - 链接：https://huggingface.co/Qwen/Qwen-7B-Chat
   - 用途：方言翻译（需要大显存 >8GB）
   - 必需度：⭐

---

## 下载方法

### 方法 1：使用下载脚本（推荐）

```bash
# 查看所有模型
python scripts/download_models.py --list

# 下载小型模型（推荐，约 900 MB）
python scripts/download_models.py --category small

# 下载小型+中型模型（约 1.7 GB）
python scripts/download_models.py --category medium

# 下载所有模型（约 15.7 GB）
python scripts/download_models.py --category all

# 下载指定模型
python scripts/download_models.py --model gpt2
```

### 方法 2：使用 Python 代码

```python
from transformers import AutoModel, AutoTokenizer

# 下载模型和 tokenizer
model_name = "uer/gpt2-chinese-cluecorpussmall"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

print(f"✓ {model_name} 下载完成")
```

### 方法 3：使用 huggingface-cli

```bash
# 安装 CLI 工具
pip install huggingface_hub

# 下载模型
huggingface-cli download uer/gpt2-chinese-cluecorpussmall

# 下载到指定目录
huggingface-cli download uer/gpt2-chinese-cluecorpussmall --local-dir ./models/
```

### 方法 4：手动下载（网络不好时）

1. 访问模型页面（如 https://huggingface.co/gpt2）
2. 点击 "Files and versions" 标签
3. 下载所有文件到本地目录
4. 使用本地路径加载：
   ```python
   model = AutoModel.from_pretrained("./models/gpt2")
   ```

---

## 使用国内镜像（推荐）

如果 HuggingFace 访问慢，可以使用国内镜像：

### 临时使用

```bash
# Linux/Mac
export HF_ENDPOINT=https://hf-mirror.com

# Windows (PowerShell)
$env:HF_ENDPOINT="https://hf-mirror.com"

# Windows (CMD)
set HF_ENDPOINT=https://hf-mirror.com

# 然后运行下载脚本
python scripts/download_models.py --category small
```

### 永久配置

在 Python 代码中添加：

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

或在 `~/.bashrc` / `~/.zshrc` 中添加：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 下载位置

模型会自动下载到：

**Windows**:
```
C:\Users\<用户名>\.cache\huggingface\hub\
```

**Linux/Mac**:
```
~/.cache/huggingface/hub/
```

### 查看已下载的模型

```bash
# Windows
dir %USERPROFILE%\.cache\huggingface\hub

# Linux/Mac
ls ~/.cache/huggingface/hub/
```

### 更改下载位置

```bash
# 设置环境变量
export HF_HOME=/path/to/your/cache
```

---

## 验证下载

下载完成后，运行测试验证：

```bash
# 测试所有项目
python scripts/run_all_tests.py

# 测试特定项目
python scripts/test_project_03.py
python scripts/test_project_04.py
```

如果看到类似错误：
```
OSError: Can't load the model for 'xxx'. If you were trying to load it from 'https://huggingface.co/models'...
```

说明模型未下载或下载不完整。

---

## 常见问题

### Q1: 下载速度很慢怎么办？

**A**: 使用国内镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q2: 下载中断了怎么办？

**A**: 重新运行下载脚本，会自动续传：
```bash
python scripts/download_models.py --category small
```

### Q3: 磁盘空间不够怎么办？

**A**:
1. 只下载小型模型（约 900 MB）
2. 或者更改下载位置到大容量磁盘：
   ```bash
   export HF_HOME=D:/huggingface_cache
   ```

### Q4: 如何删除已下载的模型？

**A**: 直接删除缓存目录：
```bash
# Windows
rmdir /s %USERPROFILE%\.cache\huggingface

# Linux/Mac
rm -rf ~/.cache/huggingface
```

### Q5: 可以离线使用吗？

**A**: 可以！下载后会自动缓存，之后可以离线使用：
```python
# 会优先使用本地缓存
model = AutoModel.from_pretrained("gpt2")

# 强制只使用本地（不尝试下载）
model = AutoModel.from_pretrained("gpt2", local_files_only=True)
```

---

## 推荐下载顺序

### 最小配置（测试用）
```bash
python scripts/download_models.py --model gpt2
python scripts/download_models.py --model uer/gpt2-chinese-cluecorpussmall
```
总大小：~900 MB

### 标准配置（推荐）
```bash
python scripts/download_models.py --category medium
```
总大小：~1.7 GB

### 完整配置（如果显存充足）
```bash
python scripts/download_models.py --category all
```
总大小：~15.7 GB

---

## 下载后的使用

下载完成后，代码会自动使用本地缓存，无需修改任何代码：

```python
# 这行代码会自动使用本地缓存
model = AutoModelForCausalLM.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
```

如果想确保只使用本地文件：

```python
model = AutoModelForCausalLM.from_pretrained(
    "uer/gpt2-chinese-cluecorpussmall",
    local_files_only=True  # 只使用本地，不尝试下载
)
```

---

**提示**: 建议先下载小型模型测试，确认可以正常使用后再下载其他模型。
