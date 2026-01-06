"""
一个依赖第三方库的示例脚本：使用 pandas 和 numpy。
运行后应打印 pandas/numpy 版本、简单数据处理结果。
"""

import sys
import platform
from datetime import datetime

import numpy as np
import pandas as pd


def main():
    print("[hello_thirdparty] 启动时间:", datetime.now().isoformat())
    print("[hello_thirdparty] Python:", sys.version.replace("\n", " "))
    print("[hello_thirdparty] 平台:", platform.platform())
    print("[hello_thirdparty] numpy:", np.__version__)
    print("[hello_thirdparty] pandas:", pd.__version__)

    # 构造一个简单的 DataFrame 并做一次计算
    df = pd.DataFrame({"a": np.arange(5), "b": np.arange(5) ** 2})
    df["c"] = df["a"] + df["b"]
    print("[hello_thirdparty] DataFrame head:\n", df.head().to_string(index=False))

    print("[hello_thirdparty] 脚本运行完成，准备退出。")
    abc

if __name__ == "__main__":
    main()

