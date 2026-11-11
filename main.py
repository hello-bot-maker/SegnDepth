import sys
from object_depth_estimator import ObjectDepthEstimator


def main():
    """
    项目核心演示入口:
    通过命令行直接运行: python main.py [物体名称] [图片路径]
    """
    target = "person"
    image_path = "examples/example1.jpg"

    if len(sys.argv) > 1:
        target = sys.argv[1]
    if len(sys.argv) > 2:
        image_path = sys.argv[2]

    # 1. 初始化类 (加载 YOLO Seg 与 Depth Anything 模型)
    estimator = ObjectDepthEstimator()

    # 2. 调用核心 run() 函数
    print(f"\n🔍 正在计算图像 '{image_path}' 中物体 '{target}' 的平均深度...")
    mean_depth = estimator.run(target_class=target, picture=image_path)

    print("\n" + "=" * 55)
    if mean_depth is not None:
        print(f"🎯 目标物体: {target}")
        print(f"📏 平均物理深度: {mean_depth:.3f} 米 ({mean_depth * 100:.1f} 厘米)")
    else:
        print(f"⚠️ 在图像中未检测到目标物体 '{target}'")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
