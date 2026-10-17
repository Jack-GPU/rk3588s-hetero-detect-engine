"""确认 PC 开发容器里的转换和训练依赖可用。"""

import torch

from rknn.api import RKNN


def main() -> None:
    import ultralytics

    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu {torch.cuda.get_device_name(0)}")
    print(f"ultralytics {ultralytics.__version__}")
    print(f"rknn {RKNN}")


if __name__ == "__main__":
    main()
