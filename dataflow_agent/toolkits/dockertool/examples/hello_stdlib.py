"""
一个不依赖第三方库的简单示例脚本，用于测试 minidocker 运行与保存镜像。

运行后应打印几行信息并正常退出（退出码 0）。
"""

import sys
import platform
from datetime import datetime


def main():
    print("[hello_stdlib] 启动时间:", datetime.now().isoformat())
    print("[hello_stdlib] Python:", sys.version.replace("\n", " "))
    print("[hello_stdlib] 平台:", platform.platform())

    # 做一个简单计算
    total = sum(i * i for i in range(10))
    print("[hello_stdlib] 计算结果 sum(i*i, i=0..9):", total)

    print("[hello_stdlib] 脚本运行完成，准备退出。")


if __name__ == "__main__":
    main()

