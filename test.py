import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

from object_depth_estimator import ObjectDepthEstimator


def run_tests():
    test_img = "examples/example1.jpg"
    assert os.path.exists(test_img), f"测试图片不存在: {test_img}"

    print("[测试启动] 实例化 ObjectDepthEstimator...")
    estimator = ObjectDepthEstimator()

    # 1. 测试支持类别列表查询
    classes = estimator.get_supported_classes()
    print(f"✅ 支持检测类别总数: {len(classes)} 类 (示例: {classes[:8]})")
    assert "person" in classes
    assert "bus" in classes

    # 2. 测试对存在的目标 (person) 计算平均深度
    print("\n--- 测试 1: estimator.run('person', picture) ---")
    depth_person = estimator.run("person", test_img)
    print(f"✅ 检测到 'person'，计算平均深度为: {depth_person:.4f} 米")
    assert isinstance(depth_person, float)
    assert 1.0 < depth_person < 15.0

    # 3. 测试对另一存在的目标 (bus) 计算平均深度
    print("\n--- 测试 2: estimator.run('bus', picture) ---")
    depth_bus = estimator.run("bus", test_img)
    print(f"✅ 检测到 'bus'，计算平均深度为: {depth_bus:.4f} 米")
    assert isinstance(depth_bus, float)
    assert depth_bus > depth_person  # 公交车在背景，深度应大于前景的人

    # 4. 测试大小写不敏感输入 ('PERSON')
    print("\n--- 测试 3: 大小写不敏感支持 ('PERSON') ---")
    depth_upper = estimator.run("PERSON", test_img)
    assert abs(depth_upper - depth_person) < 1e-5
    print(f"✅ 'PERSON' 与 'person' 结果完全一致: {depth_upper:.4f} 米")

    # 5. 测试画面中不存在的目标 (dog)
    print("\n--- 测试 4: 画面中不存在的目标 ('dog') ---")
    depth_dog = estimator.run("dog", test_img)
    print(f"✅ 画面无狗，预期返回 None: {depth_dog}")
    assert depth_dog is None

    # 6. 测试非法类别名防御 (不在 YOLO 列表中)
    print("\n--- 测试 5: 非法类别名异常防御 ---")
    try:
        estimator.run("ufo_flying_saucer", test_img)
        assert False, "非法类别未抛出 ValueError！"
    except ValueError as e:
        print(f"✅ 成功捕获非法类别异常: {e.args[0].splitlines()[0]}")

    # 7. 测试多种图像格式支持 (PIL / OpenCV)
    print("\n--- 测试 6: PIL 与 OpenCV 格式兼容性 ---")
    pil_img = Image.open(test_img)
    d_pil = estimator.run("person", pil_img)
    assert abs(d_pil - depth_person) < 1e-4

    cv_bgr = cv2.imread(test_img)
    d_cv = estimator.run("person", cv_bgr, is_bgr=True)
    assert abs(d_cv - depth_person) < 1e-3
    print("✅ PIL 与 OpenCV 图像输入格式均正常兼容解析。")

    # 8. 测试 run_detailed 详细明细方法
    print("\n--- 测试 7: run_detailed 详细明细输出 ---")
    details = estimator.run_detailed("person", test_img)
    print(f"✅ 目标: {details['target_class']}, 检出实例数: {details['count']}, 全局平均深度: {details['mean_depth']:.4f} 米")
    assert details["found"] is True
    assert details["count"] >= 1
    for inst in details["instances"]:
        print(f"   - 实例 #{inst['instance_id']}: 置信度 {inst['confidence']:.2f}, 平均深度 {inst['mean_depth']:.3f}m, 像素 {inst['pixel_count']}")

    print("\n" + "=" * 55)
    print("🎉 ObjectDepthEstimator 类的 run() 函数自测全部 100% 通过！")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    run_tests()
