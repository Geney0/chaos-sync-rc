import numpy as np
import pandas as pd
from verify_chen_buggy import compare
from pathlib import Path
print("当前工作目录：", Path.cwd())
ROOT = Path(__file__).resolve().parent.parent   # 项目根目录
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)


def to_dataframe(traj, dt):
    """把 (3, n+1) 的轨迹，转成一张表：一行一个时刻，一列一个变量。"""
    df = pd.DataFrame(traj.T, columns=["x", "y", "z"])   # ← 先转置，再起列名
    df.insert(0, "t", np.arange(len(df)) * dt)           # ← 在最左边插入时间列
    # df.insert(loc, column, value)
    return df

if __name__ == "__main__":
    result = compare([1.0, 1.0, 1.0], 0.001, 20.0) # 对应rk4()

    for name, traj in result.items():
        df = to_dataframe(traj, dt=0.001)
        path = OUT / f"{name}.csv"  # ✅ 用pathlib拼接路径
        df.to_csv(path, index=False)
        print(f"存好了：{path}，形状 {df.shape}")