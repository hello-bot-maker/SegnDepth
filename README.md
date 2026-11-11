# ObjectDepthEstimator: 物体级单目度量深度估计系统

基于 **YOLOv8-Seg (实例分割)** 与 **Depth Anything V2 (Indoor Metric 物理度量深度)** 构建的物体级深度估计系统。

输入目标物体名称（在 YOLO Seg 类别列表中）与图像，算法自动分割该物体的实例掩码，并输出其绝对物理平均深度（单位：米）。

---

## 📁 项目目录结构

```text
├── object_depth_estimator.py   # 核心实现类: ObjectDepthEstimator (包含 run() 函数)
├── main.py                     # 演示入口脚本: 支持终端参数直接运行
├── test.py                     # 自动化测试脚本: 覆盖全部核心方法与边界校验
├── examples/
│   └── example1.jpg            # 示例测试图像
├── pyproject.toml              # 项目依赖与环境配置文件
├── yolov8n-seg.pt              # YOLOv8 分割模型权重
└── README.md                   # 项目使用与提交说明文档
```

---

## 🚀 快速使用说明

### 1. Python 代码调用

```python
from object_depth_estimator import ObjectDepthEstimator

# 1. 实例化估计器类 (自动载入 YOLOv8-Seg 与 Depth Anything 模型)
estimator = ObjectDepthEstimator()

# 2. 调用核心 run() 函数
# 参数: target_class (物体名称, 如 'person', 'bus'), picture (文件路径 / PIL / numpy 矩阵)
depth = estimator.run(target_class="person", picture="examples/example1.jpg")

if depth is not None:
    print(f"目标 'person' 的平均物理深度: {depth:.3f} 米")
else:
    print("画面中未检测到该物体")
```

#### 函数规范：`estimator.run(target_class, picture, conf=0.25, is_bgr=False)`
- **`target_class`** (`str`): 目标名称（不区分大小写，如 `'person'`, `'chair'`, `'bus'`，共 80 类）；
- **`picture`** (`str` | `Path` | `PIL.Image` | `np.ndarray`): 输入图片，支持本地路径、PIL 对象或 OpenCV 矩阵；
- **`conf`** (`float`, 可选): 检测置信度阈值，默认 `0.25`；
- **`is_bgr`** (`bool`, 可选): 当输入为 OpenCV `cv2.imread` 读取的 BGR 数组时设为 `True`；
- **返回值** (`float` | `None`): 
  - 检出时返回该物体的掩码平均物理深度（单位：米）；
  - 未检出时返回 `None`；
  - 若输入的物体名称非法，抛出提示候选列表的 `ValueError`。

---

### 2. 终端命令行调用

在终端中执行：
```bash
# 语法: python main.py [物体名称] [图片路径]
uv run python main.py person examples/example1.jpg
uv run python main.py bus examples/example1.jpg
```
输出示例：
```text
🔍 正在计算图像 'examples/example1.jpg' 中物体 'person' 的平均深度...

=======================================================
🎯 目标物体: person
📏 平均物理深度: 4.440 米 (444.0 厘米)
=======================================================
```

---

### 3. 运行完整测试集

在终端中执行：
```bash
uv run python test.py
```
覆盖文件路径输入、PIL 输入、OpenCV 矩阵输入、大小写兼容、无目标防御、非法类别名校验等 7 项单元测试。
