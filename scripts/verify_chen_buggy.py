import numpy as np
from src.systems import chen
from chen_buggy import chen_buggy   # 你需要自己加这个函数

def rk4(f, v, dt, n):
    # 不是你手写的注解，不是标称类型（nominal type），是 IDE（Pylance / Pyright）的弹窗提示、鸭子类型（Protocol 结构类型）的可视化简写**，不是 Python 原生语法
    """给定向量场 f、起点 v、步长 dt，跑 n 步，返回整条轨迹。"""
    vs = np.empty((3, n + 1))
    vs[:, 0] = v
    for j in range(n):
        k1 = f(v) # 决定了f这个入参是一个callable的对象
        k2 = f(v + 0.5 * dt * k1)
        k3 = f(v + 0.5 * dt * k2)
        k4 = f(v + dt * k3)
        v = v + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        vs[:, j + 1] = v
    return vs


def simulate(f, v0, dt, T):
    """跑 T 秒（不是 n 步！），返回轨迹 (3, n+1)。"""
    n = int(round(T / dt))
    v0 = np.array(v0, dtype=float)
    # ↓ 你填：调用 rk4，把结果 return 出去
    return rk4(f, v0, dt, n)

def compare(v0, dt, T):
    """同时跑 chen 和 chen_buggy，返回一个字典。"""
    return {
        "正确": simulate(chen, v0, dt, T),      # 调用 simulate(chen, ...)
        "可疑": simulate(chen_buggy, v0, dt, T),      # 调用 simulate(chen_buggy, ...)
    }


if __name__ == "__main__":
    result = compare([1.0, 1.0, 1.0], 0.001, 20.0)
    for name, traj in result.items():
        print(f"{name}: x 范围 [{traj[0].min():.2f}, {traj[0].max():.2f}]")