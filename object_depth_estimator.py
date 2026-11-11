import os
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from transformers import pipeline
from ultralytics import YOLO

DEFAULT_YOLO_MODEL = "yolov8n-seg.pt"
DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


def _get_device() -> str:
    """自动检测硬件加速设备 (优先 GPU / Mac MPS，否则 CPU)。"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _convert_pic_to_pil(
    pic: Union[str, Path, Image.Image, np.ndarray],
    is_bgr: bool = False
) -> Image.Image:
    """
    统一将各种格式的图像输入转换为 PIL.Image.Image (RGB 格式)。
    
    支持输入类型:
        - str / Path: 本地图片路径
        - PIL.Image.Image: PIL 图像对象
        - np.ndarray: numpy 矩阵 (支持 HxWxC 格式的 RGB/BGR 或单通道灰度图)
    """
    if isinstance(pic, (str, Path)):
        pic_path = str(pic)
        if not os.path.exists(pic_path):
            raise FileNotFoundError(f"未找到指定的图像文件: '{pic_path}'")
        return Image.open(pic_path).convert("RGB")
    elif isinstance(pic, Image.Image):
        return pic.convert("RGB")
    elif isinstance(pic, np.ndarray):
        if pic.ndim == 2:
            return Image.fromarray(pic).convert("RGB")
        elif pic.ndim == 3:
            if is_bgr and pic.shape[2] == 3:
                rgb = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb)
            return Image.fromarray(pic).convert("RGB")
        else:
            raise ValueError(f"不支持的 numpy 图像维度: {pic.shape}，期望维度为 2 或 3。")
    else:
        raise TypeError(
            f"不支持的图像类型: {type(pic)}。"
            "参数 picture 支持: 文件路径 (str/Path)、PIL.Image 或 numpy.ndarray。"
        )


class ObjectDepthEstimator:
    """
    物体级单目度量深度估计器类。
    
    流水线流程:
        1. picture -> YOLO Seg : 提取指定目标的实例分割掩码 (Mask)
        2. 掩码 -> Depth : 基于 Depth Anything Indoor 物理深度图，计算该物体掩码覆盖区域的平均深度 (米)
    """

    def __init__(
        self,
        yolo_weights: str = DEFAULT_YOLO_MODEL,
        depth_model_name: str = DEFAULT_DEPTH_MODEL,
        device: Optional[str] = None
    ):
        """
        初始化估计器，预先加载 YOLO 分割模型与 Depth Anything 度量深度模型。
        """
        self.device = device if device else _get_device()
        print(f"[ObjectDepthEstimator] 载入 YOLO 分割模型: '{yolo_weights}' (设备: {self.device})...")
        self.yolo_model = YOLO(yolo_weights)
        
        print(f"[ObjectDepthEstimator] 载入深度估计模型: '{depth_model_name}' (设备: {self.device})...")
        self.depth_pipe = pipeline("depth-estimation", model=depth_model_name, device=self.device)
        self.depth_model_name = depth_model_name

        # 缓存支持的 COCO 80 类别字典 (全小写键 -> 官方原名)
        self.supported_classes = {name.lower(): name for name in self.yolo_model.names.values()}
        print(f"[ObjectDepthEstimator] 预载就绪，共支持 {len(self.supported_classes)} 类物体检测。")

    def get_supported_classes(self) -> List[str]:
        """获取模型支持检测的全部 80 种物体类别列表。"""
        return list(self.supported_classes.values())

    def _infer_depth_map(self, pil_img: Image.Image) -> np.ndarray:
        """运行单目深度推理，输出以‘米’为单位的 (H, W) float32 绝对物理深度图。"""
        width, height = pil_img.size
        res = self.depth_pipe(pil_img)
        
        if "predicted_depth" in res:
            predicted = res["predicted_depth"]
            if isinstance(predicted, torch.Tensor):
                depth_map = predicted.squeeze().detach().cpu().numpy().astype(np.float32)
            else:
                depth_map = np.array(predicted, dtype=np.float32)
        elif "depth" in res:
            depth_map = np.array(res["depth"], dtype=np.float32)
        else:
            raise ValueError("深度模型未返回有效深度输出。")

        # 确保尺寸与原图一致
        if depth_map.shape[0] != height or depth_map.shape[1] != width:
            depth_map = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_CUBIC)

        return depth_map

    def run(
        self,
        target_class: str,
        picture: Union[str, Path, Image.Image, np.ndarray],
        conf: float = 0.25,
        is_bgr: bool = False
    ) -> Optional[float]:
        """
        核心函数: 输入物体名称（在 YOLO Seg 类别列表中）与图片，输出该物体的平均物理深度（单位: 米）。

        参数:
            target_class (str): 
                目标物体名称（不区分大小写，如 'person', 'chair', 'bus', 'car' 等）。
            picture (str | Path | Image.Image | np.ndarray): 
                输入图片，支持本地文件路径、PIL 图像对象或 numpy 矩阵 (OpenCV 读取图)。
            conf (float, 可选): 
                YOLO 目标检测置信度阈值，默认 0.25。
            is_bgr (bool, 可选): 
                若传入的 numpy 数组为 OpenCV 原生 BGR 格式，请设为 True（默认 False 为 RGB）。

        返回:
            float | None: 
                - 若在图片中检测到该物体，返回该物体掩码区域内像素的平均物理深度（单位: 米）；
                - 若画面中未检测到该物体，返回 None。

        异常:
            ValueError: 当输入的 target_class 不在 YOLO 模型支持的 80 类列表中时抛出。
            FileNotFoundError: 当 picture 为不存在的文件路径时抛出。
            TypeError: 当 picture 为非法类型时抛出。
        """
        # 1. 目标类别校验
        target_key = target_class.strip().lower()
        if target_key not in self.supported_classes:
            valid_sample = list(self.supported_classes.values())[:15]
            raise ValueError(
                f"错误：'{target_class}' 不在 YOLO 分割模型支持的类别列表中！\n"
                f"常见类别示例: {valid_sample} ... (共 80 类，可通过 get_supported_classes() 查询完整列表)"
            )
        official_class_name = self.supported_classes[target_key]

        # 2. 图像标准化解析
        pil_img = _convert_pic_to_pil(picture, is_bgr=is_bgr)
        width, height = pil_img.size

        # 3. 步骤 1: picture -> YOLO Seg 获取目标分割掩码
        yolo_res = self.yolo_model(pil_img, conf=conf, verbose=False)[0]

        if yolo_res.masks is None or len(yolo_res.masks) == 0:
            print(f"[ObjectDepthEstimator] 画面中未检测到任何具备掩码的目标。")
            return None

        # 过滤出属于 target_class 的目标实例
        boxes = yolo_res.boxes
        target_indices = []
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            cls_name = yolo_res.names[cls_id]
            if cls_name.lower() == target_key:
                target_indices.append(i)

        if not target_indices:
            print(f"[ObjectDepthEstimator] 画面中未检测到目标物体 '{official_class_name}' (置信度: {conf})。")
            return None

        # 4. 步骤 2: 掩码对齐与平均物理深度计算
        # 将 YOLO 掩码双线性插值到原图尺寸 (H, W)
        masks_tensor = yolo_res.masks.data[target_indices].unsqueeze(1)
        masks_resized = F.interpolate(
            masks_tensor,
            size=(height, width),
            mode="bilinear",
            align_corners=False
        ).squeeze(1)
        binary_masks = (masks_resized > 0.5).cpu().numpy().astype(bool)

        # 合并目标的所有实例掩码
        combined_target_mask = np.any(binary_masks, axis=0)

        # 推理获取全局物理绝对深度图 (单位: 米)
        depth_map = self._infer_depth_map(pil_img)

        # 提取目标掩码覆盖区域内的所有有效深度像素
        target_depths = depth_map[combined_target_mask]
        valid_depths = target_depths[np.isfinite(target_depths) & (target_depths > 0)]

        if len(valid_depths) == 0:
            return None

        # 计算并返回平均深度 (米)
        mean_depth = float(np.mean(valid_depths))
        return mean_depth

    def run_detailed(
        self,
        target_class: str,
        picture: Union[str, Path, Image.Image, np.ndarray],
        conf: float = 0.25,
        is_bgr: bool = False
    ) -> Dict[str, Any]:
        """
        扩展方法: 返回详细分析报告（包含检出数量、各实例明细、全图深度矩阵与标注图）。
        """
        target_key = target_class.strip().lower()
        if target_key not in self.supported_classes:
            raise ValueError(f"'{target_class}' 不是 YOLO 支持的有效类别名。")
        official_class_name = self.supported_classes[target_key]

        pil_img = _convert_pic_to_pil(picture, is_bgr=is_bgr)
        orig_np = np.array(pil_img)
        width, height = pil_img.size

        yolo_res = self.yolo_model(pil_img, conf=conf, verbose=False)[0]
        depth_map = self._infer_depth_map(pil_img)

        if yolo_res.masks is None or len(yolo_res.masks) == 0:
            return {
                "found": False, "target_class": official_class_name, "count": 0,
                "mean_depth": None, "instances": [], "depth_map": depth_map, "annotated_image": orig_np
            }

        boxes = yolo_res.boxes
        target_indices = [i for i in range(len(boxes)) if yolo_res.names[int(boxes.cls[i].item())].lower() == target_key]

        if not target_indices:
            return {
                "found": False, "target_class": official_class_name, "count": 0,
                "mean_depth": None, "instances": [], "depth_map": depth_map, "annotated_image": orig_np
            }

        masks_tensor = yolo_res.masks.data[target_indices].unsqueeze(1)
        masks_resized = F.interpolate(
            masks_tensor, size=(height, width), mode="bilinear", align_corners=False
        ).squeeze(1)
        binary_masks = (masks_resized > 0.5).cpu().numpy().astype(bool)

        instances = []
        for idx, orig_i in enumerate(target_indices):
            mask = binary_masks[idx]
            conf_val = float(boxes.conf[orig_i].item())
            bbox = boxes.xyxy[orig_i].cpu().numpy().astype(int).tolist()
            d_vals = depth_map[mask]
            valid_d = d_vals[np.isfinite(d_vals) & (d_vals > 0)]

            inst_mean = float(np.mean(valid_d)) if len(valid_d) > 0 else 0.0
            inst_median = float(np.median(valid_d)) if len(valid_d) > 0 else 0.0
            instances.append({
                "instance_id": idx + 1,
                "confidence": conf_val,
                "bbox": bbox,
                "mean_depth": inst_mean,
                "median_depth": inst_median,
                "pixel_count": int(np.sum(mask)),
                "mask": mask
            })

        combined_mask = np.any(binary_masks, axis=0)
        overall_valid = depth_map[combined_mask]
        overall_valid = overall_valid[np.isfinite(overall_valid) & (overall_valid > 0)]
        overall_mean = float(np.mean(overall_valid)) if len(overall_valid) > 0 else None

        # 标注可视化图
        canvas = orig_np.copy()
        for inst in instances:
            canvas[inst["mask"]] = (canvas[inst["mask"]] * 0.55 + np.array([0, 220, 255]) * 0.45).astype(np.uint8)
            x1, y1, x2, y2 = inst["bbox"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 255), 2)
            tag = f"{official_class_name} #{inst['instance_id']}: {inst['mean_depth']:.2f}m"
            cv2.putText(canvas, tag, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return {
            "found": True,
            "target_class": official_class_name,
            "count": len(instances),
            "mean_depth": overall_mean,
            "instances": instances,
            "depth_map": depth_map,
            "annotated_image": canvas,
        }
