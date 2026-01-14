# MinerU 技术实现方案

本文档详细介绍 MinerU 的技术实现方案，按照由粗到细的顺序阐述系统的设计与实现。

## 一、项目概述

### 1.1 项目定位

MinerU 是一款高精度的 PDF 文档内容提取工具，能够将 PDF 文档转化为机器可读格式（如 Markdown、JSON 等），诞生于书生-浦语大模型的预训练过程中。

### 1.2 核心能力

- **版面分析**：识别页眉、页脚、脚注、页码等元素
- **阅读顺序**：输出符合人类阅读习惯的文本序列
- **结构保留**：保留标题、段落、列表等文档结构
- **多元素提取**：提取图像、表格、公式等元素
- **格式转换**：公式转 LaTeX，表格转 HTML
- **OCR 支持**：支持扫描版 PDF 和 109 种语言识别
- **多平台兼容**：支持 Windows/Linux/macOS，支持 CPU/GPU/NPU/MPS 加速

## 二、系统架构

### 2.1 整体架构

MinerU 采用模块化的分层架构设计，主要包含以下层次：

```
┌────────────────────────────────────────────────────────────┐
│                      用户接口层                              │
│     CLI | Python API | WebUI (Gradio) | FastAPI Server      │
├────────────────────────────────────────────────────────────┤
│                      解析后端层                              │
│         pipeline | vlm | hybrid-auto-engine                 │
├────────────────────────────────────────────────────────────┤
│                      模型服务层                              │
│   Layout | MFD | MFR | OCR | Table | VLM | Reading Order   │
├────────────────────────────────────────────────────────────┤
│                      数据处理层                              │
│     PDF Reader | Image Tools | Data Writer | Utilities      │
└────────────────────────────────────────────────────────────┘
```

### 2.2 核心目录结构

```
mineru/
├── backend/           # 解析后端实现
│   ├── pipeline/      # 传统 pipeline 后端
│   ├── vlm/           # VLM 视觉语言模型后端
│   └── hybrid/        # 混合后端
├── cli/               # 命令行工具
├── data/              # 数据读写模块
├── model/             # AI 模型封装
│   ├── layout/        # 版面分析模型
│   ├── mfd/           # 公式检测模型
│   ├── mfr/           # 公式识别模型
│   ├── ocr/           # OCR 识别模型
│   ├── table/         # 表格识别模型
│   ├── reading_order/ # 阅读顺序模型
│   └── vlm/           # VLM 推理服务
├── resources/         # 资源文件
└── utils/             # 工具函数
```

## 三、解析后端详解

MinerU 提供三种解析后端，满足不同场景需求：

### 3.1 Pipeline 后端

传统的多模型级联架构，兼容性最好，支持纯 CPU 环境。

**处理流程**：

```
PDF输入 → 页面图像化 → 版面分析(Layout) → 元素分类
                              ↓
            ┌─────────────────┼─────────────────┐
            ↓                 ↓                 ↓
       文本区域           公式区域           表格区域
            ↓                 ↓                 ↓
         OCR识别         公式检测+识别       表格识别
            ↓                 ↓                 ↓
            └─────────────────┼─────────────────┘
                              ↓
                       阅读顺序排序
                              ↓
                       输出结构化结果
```

**核心模块**：

- **DocLayout-YOLO**：版面分析，检测文本、图片、表格、公式等区域
- **YOLOv8-MFD**：公式检测（Math Formula Detection）
- **UniMERNet / PP-FormulaNet**：公式识别（Math Formula Recognition）
- **PaddleOCR**：文本识别
- **SLANet / UNet-Table**：表格结构识别

**适用场景**：
- 对硬件要求低的环境
- 需要高度定制化的场景
- 批量处理大量文档

### 3.2 VLM 后端

基于视觉语言模型（Vision-Language Model）的端到端解析方案。

**处理流程**：

```
PDF输入 → 页面图像化 → VLM推理(两阶段) → 结构化输出
                              ↓
                    Stage 1: 版面检测
                              ↓
                    Stage 2: 内容识别
                              ↓
                       输出结构化结果
```

**技术特点**：

- 采用 Qwen2-VL 系列模型
- 支持多种推理引擎：transformers、vLLM、LMDeploy、MLX
- 端到端设计，减少级联误差

**推理引擎支持**：

| 引擎 | 特点 | 适用场景 |
|------|------|----------|
| transformers | 原生实现，易于调试 | 开发测试 |
| vllm-engine | 高性能，支持连续批处理 | 生产部署 |
| lmdeploy-engine | 华为昇腾适配 | NPU环境 |
| mlx-engine | Apple Silicon 优化 | macOS |
| http-client | 远程调用，无需本地GPU | C/S架构 |

### 3.3 Hybrid 后端

结合 Pipeline 和 VLM 优势的混合架构（推荐使用）。

**设计理念**：

```
PDF输入 → PDF分类(文本型/扫描型)
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
 文本型PDF           扫描型PDF
    ↓                   ↓
VLM版面识别        VLM版面识别
    ↓                   ↓
 PDF文本抽取        OCR文本识别
    ↓                   ↓
    └─────────┬─────────┘
              ↓
       行内公式检测+识别(可选)
              ↓
         输出结构化结果
```

**核心优势**：

- 文本 PDF：直接从 PDF 抽取文本，原生支持多语言，减少 OCR 幻觉
- 扫描 PDF：结合 OCR 支持 109 种语言
- 行内公式：独立开关控制，按需启用

## 四、核心模型详解

### 4.1 版面分析模型（Layout）

基于 DocLayout-YOLO（YOLOv10 变体）实现。

**类别定义**：

| ID | 类别名称 | 说明 |
|----|----------|------|
| 0 | title | 标题 |
| 1 | plain text | 正文文本 |
| 2 | abandon | 页眉页脚等丢弃内容 |
| 3 | figure | 图片 |
| 4 | figure_caption | 图片标题 |
| 5 | table | 表格 |
| 6 | table_caption | 表格标题 |
| 7 | table_footnote | 表格脚注 |
| 8 | isolate_formula | 独立公式 |
| 9 | formula_caption | 公式编号 |

**实现细节**：

```python
class DocLayoutYOLOModel:
    def __init__(self, weight, device="cuda", imgsz=1280, conf=0.1, iou=0.45):
        self.model = YOLOv10(weight).to(device)
        
    def predict(self, image):
        prediction = self.model.predict(image, imgsz=self.imgsz, ...)
        return self._parse_prediction(prediction)
```

### 4.2 公式检测模型（MFD）

基于 YOLOv8 的数学公式检测模型。

**检测类型**：
- 行内公式（Inline）
- 行间公式（Interline）

### 4.3 公式识别模型（MFR）

支持两种公式识别模型：

**UniMERNet**（默认）：
- 基于 Encoder-Decoder 架构
- 输出 LaTeX 格式
- 支持复杂公式

**PP-FormulaNet-Plus-M**：
- 来自 PaddleOCR
- 更好的中文公式支持
- 通过 `MINERU_FORMULA_CH_SUPPORT=True` 启用

### 4.4 OCR 模型

封装 PaddleOCR 的 PyTorch 版本。

**核心功能**：
- 文本检测（DBNet）
- 文本识别（CRNN/SVTR）
- 支持 109 种语言

**实现架构**：

```python
class PytorchPaddleOCR:
    def __init__(self, lang="ch", ...):
        self.text_detector = TextDetector(...)
        self.text_recognizer = TextRecognizer(...)
        
    def ocr(self, images, det=True, rec=True):
        if det:
            boxes = self.text_detector(images)
        if rec:
            texts = self.text_recognizer(images)
        return results
```

### 4.5 表格识别模型

**表格分类**：
- PaddleTableClsModel：判断有线表/无线表

**表格结构识别**：
- UnetTableModel：有线表格
- RapidTableModel (SLANet-Plus)：无线表格

**输出格式**：HTML 表格结构

### 4.6 阅读顺序模型

提供两种阅读顺序算法：

**XY-Cut 算法**：
- 递归投影分割
- 基于规则的排序
- 轻量级，无需 GPU

**LayoutReader**：
- 基于 LayoutLMv3
- 学习排序策略
- 更准确但需要 GPU

```python
def recursive_xy_cut(boxes, indices, res):
    """XY-Cut 递归投影分割算法"""
    # 1. Y轴投影，水平切分
    y_projection = projection_by_bboxes(boxes, axis=1)
    # 2. X轴投影，垂直切分
    x_projection = projection_by_bboxes(boxes, axis=0)
    # 3. 递归处理子区域
```

## 五、数据处理流程

### 5.1 PDF 处理

**PDF 分类**：

```python
def classify(pdf_bytes) -> str:
    """判断 PDF 类型：文本型或扫描型"""
    # 分析文本层密度
    # 返回 'txt' 或 'ocr'
```

**页面图像化**：

```python
def load_images_from_pdf(pdf_bytes, image_type):
    """将 PDF 页面转换为图像"""
    # 使用 pypdfium2 渲染
    # 支持 PIL/NumPy 格式输出
```

### 5.2 中间 JSON 格式

MinerU 使用统一的中间 JSON 格式表示解析结果：

```json
{
    "pdf_info": [
        {
            "page_idx": 0,
            "page_size": [595, 842],
            "para_blocks": [
                {
                    "type": "text",
                    "bbox": [x0, y0, x1, y1],
                    "lines": [
                        {
                            "spans": [
                                {"type": "text", "content": "..."}
                            ]
                        }
                    ]
                }
            ],
            "discarded_blocks": []
        }
    ],
    "_backend": "hybrid",
    "_version_name": "2.7.1"
}
```

### 5.3 输出格式转换

中间 JSON 可转换为多种输出格式：

- **Markdown**：带图片引用的 Markdown 文本
- **Content List JSON**：按阅读顺序的内容列表
- **Structured JSON**：完整结构化信息

## 六、模型管理与优化

### 6.1 单例模式管理

所有模型采用单例模式管理，避免重复加载：

```python
class ModelSingleton:
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self, key, **kwargs):
        if key not in self._models:
            self._models[key] = self._init_model(**kwargs)
        return self._models[key]
```

### 6.2 批处理优化

支持动态批处理以提升性能：

```python
def get_batch_ratio(device):
    """根据显存自动计算批处理比例"""
    gpu_memory = get_vram(device)
    if gpu_memory >= 32:
        return 16
    elif gpu_memory >= 16:
        return 8
    elif gpu_memory >= 12:
        return 4
    elif gpu_memory >= 8:
        return 2
    return 1
```

### 6.3 内存管理

主动进行内存清理：

```python
def clean_memory(device):
    """清理 GPU/NPU 内存"""
    import gc
    gc.collect()
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    elif str(device).startswith('npu'):
        torch_npu.npu.empty_cache()
```

## 七、部署架构

### 7.1 单机部署

标准的单机部署模式：

```bash
# 安装
uv pip install -U "mineru[all]"

# 命令行使用
mineru -p input.pdf -o output/

# 指定后端
mineru -p input.pdf -o output/ -b pipeline
```

### 7.2 C/S 架构部署

支持 Server/Client 分离部署：

**Server 端**（GPU 服务器）：

```bash
# 启动 VLM 推理服务
python -m mineru.model.vlm.vllm_server --model-path /path/to/model
```

**Client 端**（CPU 机器）：

```bash
# 连接远程服务
mineru -p input.pdf -o output/ -b vlm-http-client --server-url http://server:8000
```

### 7.3 Docker 部署

提供预构建的 Docker 镜像：

```bash
docker pull opendatalab/mineru:latest
docker run -v /data:/data mineru -p /data/input.pdf -o /data/output/
```

## 八、扩展机制

### 8.1 自定义模型

支持替换内置模型：

```python
# 自定义版面分析模型
class CustomLayoutModel:
    def predict(self, image):
        # 自定义实现
        return results
```

### 8.2 LLM 辅助优化

支持 LLM 辅助优化，如标题分级：

```python
# 配置 LLM 辅助
llm_aided_config = {
    "title_aided": {
        "enable": True,
        "model": "gpt-4",
        "api_key": "..."
    }
}
```

## 九、性能指标

### 9.1 精度对比

| 后端 | OmniDocBench 分数 |
|------|------------------|
| pipeline | 82+ |
| vlm | 90+ |
| hybrid | 90+ |

### 9.2 硬件要求

| 后端 | 显存要求 | 内存要求 |
|------|---------|---------|
| pipeline | 6GB+ | 16GB+ |
| hybrid-auto-engine | 10GB+ | 16GB+ |
| vlm-auto-engine | 8GB+ | 16GB+ |
| *-http-client | 不需要 | 8GB+ |

## 十、总结

MinerU 通过模块化的架构设计，提供了从传统 Pipeline 到端到端 VLM 的多种解析方案，用户可以根据实际场景选择合适的后端。系统的核心优势在于：

1. **高精度**：VLM/Hybrid 后端在 OmniDocBench 上达到 90+ 分数
2. **灵活性**：三种后端满足不同硬件和精度需求
3. **易用性**：简单的命令行和 API 接口
4. **可扩展**：模块化设计支持自定义扩展

更多技术细节请参考：
- [技术报告 MinerU](https://arxiv.org/abs/2409.18839)
- [技术报告 MinerU2.5](https://arxiv.org/abs/2509.22186)
- [API 文档](./index.md)
