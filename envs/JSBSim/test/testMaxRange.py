import numpy as np
import matplotlib.pyplot as plt


def calculate_missile_range(
        g=9.81,  # 重力加速度 m/s^2
        t_max=60,  # 导弹最大飞行时间 s
        t_thrust=3,  # 发动机工作时间 s
        Isp=120,  # 比冲 s
        Length=2.87,  # 长度 m
        Diameter=0.127,  # 直径 m
        cD=0.4,  # 阻力系数
        m0=84,  # 初始质量 kg
        dm=6,  # 质量损失率 kg/s
        K=3,  # 比例导航常数
        nyz_max=30,  # 最大过载 g
        Rc=300,  # 爆炸半径 m
        v_min=150,  # 最小速度 m/s
        v_initial=300,  # 初始发射速度 m/s
        launch_angle=0,  # 发射角度 度
        altitude=0,  # 发射高度 m
        dt=0.1,  # 计算时间步长 s
        verbose=True  # 是否输出详细信息
):
    """
    计算导弹的最大射程

    参数说明:
    - g: 重力加速度 (m/s^2)
    - t_max: 导弹最大飞行时间 (s)
    - t_thrust: 发动机工作时间 (s)
    - Isp: 比冲 (s)
    - Length: 导弹长度 (m)
    - Diameter: 导弹直径 (m)
    - cD: 阻力系数
    - m0: 初始质量 (kg)
    - dm: 质量损失率 (kg/s)
    - K: 比例导航常数
    - nyz_max: 最大过载 (g)
    - Rc: 爆炸半径 (m)
    - v_min: 最小速度 (m/s)
    - v_initial: 初始发射速度 (m/s)
    - launch_angle: 发射角度 (度)
    - altitude: 发射高度 (m)
    - dt: 计算时间步长 (s)
    - verbose: 是否输出详细信息

    返回:
    - dict: 包含射程、速度等详细信息的字典
    """

    # 转换角度为弧度
    launch_angle_rad = np.radians(launch_angle)

    # 计算基本参数
    S = np.pi * (Diameter / 2) ** 2  # 横截面积
    thrust = g * Isp * dm  # 推力
    m_burnout = m0 - dm * t_thrust  # 燃烧结束时质量

    # 空气密度计算（考虑高度影响）
    def air_density(h):
        """计算给定高度的空气密度"""
        rho0 = 1.225  # 海平面空气密度
        if h <= 11000:  # 对流层
            T = 288.15 - 0.0065 * h
            return rho0 * (T / 288.15) ** 4.25588
        else:  # 简化的平流层
            return 0.364 * np.exp((11000 - h) / 6341.62)

    # 初始条件
    v_x = v_initial * np.cos(launch_angle_rad)  # 水平速度分量
    v_z = v_initial * np.sin(launch_angle_rad)  # 垂直速度分量
    x, z = 0, altitude  # 初始位置

    # 存储轨迹数据
    trajectory = {
        'time': [0],
        'x': [x],
        'z': [z],
        'v_total': [v_initial],
        'v_x': [v_x],
        'v_z': [v_z]
    }

    # 第一阶段：推进阶段
    t = 0
    m_current = m0

    if verbose:
        print("=== 推进阶段 ===")

    while t < t_thrust:
        # 当前总速度和角度
        v_total = np.sqrt(v_x ** 2 + v_z ** 2)
        flight_angle = np.arctan2(v_z, v_x)

        # 空气密度
        rho = air_density(max(0, z))

        # 阻力
        drag = 0.5 * cD * S * rho * v_total ** 2

        # 推力分量
        thrust_x = thrust * np.cos(flight_angle) - drag * np.cos(flight_angle)
        thrust_z = thrust * np.sin(flight_angle) - drag * np.sin(flight_angle)

        # 加速度
        ax = thrust_x / m_current
        az = thrust_z / m_current - g

        # 更新速度和位置
        v_x += ax * dt
        v_z += az * dt
        x += v_x * dt
        z += v_z * dt

        # 更新质量和时间
        m_current -= dm * dt
        t += dt

        # 记录轨迹
        trajectory['time'].append(t)
        trajectory['x'].append(x)
        trajectory['z'].append(max(0, z))  # 不允许负高度
        trajectory['v_total'].append(np.sqrt(v_x ** 2 + v_z ** 2))
        trajectory['v_x'].append(v_x)
        trajectory['v_z'].append(v_z)

        # 检查是否撞地
        if z < 0:
            break

    range_boost = x
    v_burnout = np.sqrt(v_x ** 2 + v_z ** 2)

    if verbose:
        print(f"推进结束时间: {t:.2f} 秒")
        print(f"推进结束速度: {v_burnout:.0f} m/s")
        print(f"推进阶段射程: {range_boost / 1000:.2f} km")
        print(f"推进结束高度: {z:.0f} m")

    # 第二阶段：滑翔阶段
    if verbose:
        print("\n=== 滑翔阶段 ===")

    glide_start_time = t
    m_current = m_burnout

    while t < t_max and z >= 0 and v_burnout > 150:
        # 当前总速度
        v_total = np.sqrt(v_x ** 2 + v_z ** 2)

        # 检查最小速度限制
        if v_total < v_min:
            break

        # 空气密度
        rho = air_density(max(0, z))

        # 阻力
        drag = 0.5 * cD * S * rho * v_total ** 2

        # 阻力分量
        if v_total > 0:
            drag_x = drag * (v_x / v_total)
            drag_z = drag * (v_z / v_total)
        else:
            drag_x = drag_z = 0

        # 加速度（只有阻力和重力）
        ax = -drag_x / m_current
        az = -drag_z / m_current - g

        # 更新速度和位置
        v_x += ax * dt
        v_z += az * dt
        x += v_x * dt
        z += v_z * dt

        t += dt

        # 记录轨迹
        trajectory['time'].append(t)
        trajectory['x'].append(x)
        trajectory['z'].append(max(0, z))
        trajectory['v_total'].append(v_total)
        trajectory['v_x'].append(v_x)
        trajectory['v_z'].append(v_z)

        # 检查是否撞地
        if z <= 0:
            break

    range_glide = x - range_boost
    final_speed = np.sqrt(v_x ** 2 + v_z ** 2)
    glide_time = t - glide_start_time

    if verbose:
        print(f"滑翔时间: {glide_time:.2f} 秒")
        print(f"滑翔结束速度: {final_speed:.0f} m/s")
        print(f"滑翔阶段射程: {range_glide / 1000:.2f} km")
        print(f"最终高度: {z:.0f} m")
        print(f"\n=== 总结 ===")
        print(f"总飞行时间: {t:.2f} 秒")
        print(f"总射程: {x / 1000:.2f} km")

    # 返回详细结果
    result = {
        'total_range_km': x / 1000,
        'boost_range_km': range_boost / 1000,
        'glide_range_km': range_glide / 1000,
        'total_time': t,
        'boost_time': t_thrust,
        'glide_time': glide_time,
        'burnout_speed': v_burnout,
        'final_speed': final_speed,
        'max_altitude': max(trajectory['z']),
        'trajectory': trajectory,
        'parameters': {
            'g': g, 't_max': t_max, 't_thrust': t_thrust, 'Isp': Isp,
            'Length': Length, 'Diameter': Diameter, 'cD': cD, 'm0': m0,
            'dm': dm, 'K': K, 'nyz_max': nyz_max, 'Rc': Rc, 'v_min': v_min,
            'v_initial': v_initial, 'launch_angle': launch_angle, 'altitude': altitude
        }
    }

    return result


def plot_trajectory(result, title="导弹飞行轨迹"):
    """绘制导弹飞行轨迹图"""
    trajectory = result['trajectory']

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # 轨迹图
    ax1.plot(np.array(trajectory['x']) / 1000, np.array(trajectory['z']) / 1000)
    ax1.set_xlabel('水平距离 (km)')
    ax1.set_ylabel('高度 (km)')
    ax1.set_title('飞行轨迹')
    ax1.grid(True)

    # 速度-时间图
    ax2.plot(trajectory['time'], trajectory['v_total'])
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('速度 (m/s)')
    ax2.set_title('速度变化')
    ax2.grid(True)

    # 水平速度分量
    ax3.plot(trajectory['time'], trajectory['v_x'], label='水平速度')
    ax3.plot(trajectory['time'], trajectory['v_z'], label='垂直速度')
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('速度分量 (m/s)')
    ax3.set_title('速度分量')
    ax3.legend()
    ax3.grid(True)

    # 高度-时间图
    ax4.plot(trajectory['time'], np.array(trajectory['z']) / 1000)
    ax4.set_xlabel('时间 (s)')
    ax4.set_ylabel('高度 (km)')
    ax4.set_title('高度变化')
    ax4.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def compare_parameters(base_params, variations, param_name):
    """比较不同参数值对射程的影响"""
    results = []

    # 参数名映射表，确保参数名与函数参数一致
    param_mapping = {
        'v_initial': 'v_initial',
        'launch_angle': 'launch_angle',
        't_thrust': 't_thrust',
        'Isp': 'Isp',  # 注意大写
        'isp': 'Isp',  # 处理小写情况
        'cD': 'cD',
        'cd': 'cD',
        'm0': 'm0',
        'dm': 'dm',
        't_max': 't_max',
        'v_min': 'v_min',
        'diameter': 'Diameter',
        'Diameter': 'Diameter',
        'length': 'Length',
        'Length': 'Length',
        'K': 'K',
        'k': 'K',
        'nyz_max': 'nyz_max',
        'Rc': 'Rc',
        'rc': 'Rc'
    }

    # 获取正确的参数名
    correct_param = param_mapping.get(param_name, param_name)

    print(f"\n=== {param_name} 对射程的影响 ===")
    print(f"{'参数值':<10} {'总射程(km)':<12} {'推进射程(km)':<14} {'滑翔射程(km)':<14}")
    print("-" * 60)

    for value in variations:
        params = base_params.copy()
        params[correct_param] = value
        params['verbose'] = False

        result = calculate_missile_range(**params)
        results.append((value, result))

        print(f"{value:<10} {result['total_range_km']:<12.2f} "
              f"{result['boost_range_km']:<14.2f} {result['glide_range_km']:<14.2f}")

    return results


# 使用示例
if __name__ == "__main__":
    # 默认参数（您提供的AIM-9L参数）
    default_params = {
        'g': 9.81,
        't_max': 60,
        't_thrust': 3,
        'Isp': 120,
        'Length': 2.87,
        'Diameter': 0.127,
        'cD': 0.4,
        'm0': 84,
        'dm': 6,
        'K': 3,
        'nyz_max': 30,
        'Rc': 300,
        'v_min': 150,
        'v_initial': 300,
        'launch_angle': 0,
        'altitude': 6000
    }

    # 计算默认参数下的射程
    print("计算默认参数下的导弹射程：")
    result = calculate_missile_range(**default_params)

    # 参数敏感性分析
    print("\n" + "=" * 60)
    print("参数敏感性分析")
    print("=" * 60)

    # 比较不同初始速度
    compare_parameters(default_params, [200, 250, 300, 350, 400, 500], 'v_initial')

    # 比较不同发射角度
    compare_parameters(default_params, [0, 10, 20, 30, 45], 'launch_angle')

    # 比较不同发动机工作时间
    compare_parameters(default_params, [2, 3, 4, 5, 6], 't_thrust')

    # 比较不同比例导航常数
    # compare_parameters(default_params, [1, 2, 3, 4, 5], 'K')
    #
    # # 比较不同最大过载
    # compare_parameters(default_params, [20, 25, 30, 35, 40], 'nyz_max')