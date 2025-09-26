import os
import yaml
import pymap3d
import numpy as np


def parse_config(filename):
    """Parse JSBSim config file.

    Args:
        config (str): config file name

    Returns:
        (EnvConfig): a custom class which parsing dict into object.
    """
    filepath = os.path.join(get_root_dir(), 'configs', f'{filename}.yaml')
    assert os.path.exists(filepath), \
        f'config path {filepath} does not exist. Please pass in a string that represents the file path to the config yaml.'
    with open(filepath, 'r', encoding='utf-8') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    return type('EnvConfig', (object,), config_data)


def get_root_dir():
    return os.path.join(os.path.split(os.path.realpath(__file__))[0], '..')


def LLA2NEU(lon, lat, alt, lon0=120.0, lat0=60.0, alt0=0):
    """Convert from Geodetic Coordinate System to NEU Coordinate System.

    Args:
        lon, lat, alt (float): target geodetic lontitude(°), latitude(°), altitude(m)
        lon, lat, alt (float): observer geodetic lontitude(°), latitude(°), altitude(m); Default=`(120°E, 60°N, 0m)`

    Returns:
        (np.array): (North, East, Up), unit: m
    """
    n, e, d = pymap3d.geodetic2ned(lat, lon, alt, lat0, lon0, alt0)
    return np.array([n, e, -d])


def NEU2LLA(n, e, u, lon0=120.0, lat0=60.0, alt0=0):
    """Convert from NEU Coordinate System to Geodetic Coordinate System.

    Args:
        n, e, u (float): target relative position w.r.t. North, East, Down
        lon, lat, alt (float): observer geodetic lontitude(°), latitude(°), altitude(m); Default=`(120°E, 60°N, 0m)`

    Returns:
        (np.array): (lon, lat, alt), unit: °, °, m
    """
    lat, lon, h = pymap3d.ned2geodetic(n, e, -u, lat0, lon0, alt0)
    return np.array([lon, lat, h])


def get_AO_TA_R(ego_feature, enm_feature, return_side=False):
    """Get AO & TA angles and relative distance between two agent.

    Args:
        ego_feature & enemy_feature (tuple): (north, east, down, vn, ve, vd)

    Returns:
        (tuple): ego_AO, ego_TA, R
    """
    ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
    ego_v = np.linalg.norm([ego_vx, ego_vy, ego_vz])
    enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enm_feature
    enm_v = np.linalg.norm([enm_vx, enm_vy, enm_vz])
    delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
    R = np.linalg.norm([delta_x, delta_y, delta_z])

    proj_dist = delta_x * ego_vx + delta_y * ego_vy + delta_z * ego_vz
    ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
    proj_dist = delta_x * enm_vx + delta_y * enm_vy + delta_z * enm_vz
    ego_TA = np.arccos(np.clip(proj_dist / (R * enm_v + 1e-8), -1, 1))

    if not return_side:
        return ego_AO, ego_TA, R
    else:
        side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
        return ego_AO, ego_TA, R, side_flag


def get2d_AO_TA_R(ego_feature, enm_feature, return_side=False):
    ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
    ego_v = np.linalg.norm([ego_vx, ego_vy])
    enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enm_feature
    enm_v = np.linalg.norm([enm_vx, enm_vy])
    delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
    R = np.linalg.norm([delta_x, delta_y])

    proj_dist = delta_x * ego_vx + delta_y * ego_vy
    ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
    proj_dist = delta_x * enm_vx + delta_y * enm_vy
    ego_TA = np.arccos(np.clip(proj_dist / (R * enm_v + 1e-8), -1, 1))

    if not return_side:
        return ego_AO, ego_TA, R
    else:
        side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
        return ego_AO, ego_TA, R, side_flag


def in_range_deg(angle):
    """ Given an angle in degrees, normalises in (-180, 180] """
    angle = angle % 360
    if angle > 180:
        angle -= 360
    return angle


def in_range_rad(angle):
    """ Given an angle in rads, normalises in (-pi, pi] """
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle

def get_az_el_R(ego_feature, enm_feature):
    """计算敌机相对于我机的方位角(az)、俯仰角(el)和距离(R)。

    该函数将全局NED坐标转换为我机的机体坐标系来进行计算。

    Args:
        ego_feature & enm_feature (tuple): (north, east, down, vn, ve, vd)

    Returns:
        (tuple): azimuth, elevation, R (角度单位为弧度 radians)
    """
    # --- 1. 解包数据并转换为numpy数组，便于向量运算 ---
    ego_pos = np.array(ego_feature[:3])
    ego_vel = np.array(ego_feature[3:])
    enm_pos = np.array(enm_feature[:3])

    # --- 2. 计算从我机指向敌机的相对位置向量 ---
    # 这是在全局NED坐标系下的向量
    delta_p_global = enm_pos - ego_pos

    # --- 3. 计算距离 R ---
    # 距离是一个标量，与坐标系无关
    R = np.linalg.norm(delta_p_global)
    if R < 1e-6: # 避免距离过近导致的计算错误
        return 0.0, 0.0, R

    # --- 4. 构建我机的机体坐标系 (Body Frame) ---
    # 使用速度矢量来确定机头朝向

    # a. 机身X轴 (前向) 是归一化的速度矢量
    ego_v_norm = np.linalg.norm(ego_vel)
    x_body = ego_vel / (ego_v_norm + 1e-8)  # +1e-8 防止除以零

    # b. 机身Y轴 (右向) 是前向矢量与全局“下”矢量的叉积
    # 在NED坐标系中, 全局“下”矢量是 [0, 0, 1]
    global_down = np.array([0., 0., 1.])
    # 叉积结果垂直于x_body和global_down，指向右翼
    y_body_unnormalized = np.cross(x_body, global_down)
    y_body_norm = np.linalg.norm(y_body_unnormalized)
    y_body = y_body_unnormalized / (y_body_norm + 1e-8)

    # c. 机身Z轴 (下向) 是前向与右向的叉积，完成右手坐标系
    z_body = np.cross(x_body, y_body)

    # --- 5. 将全局的相对位置向量投影到机体坐标系上 ---
    # 这通过与每个机身轴做点积来实现
    x_local = np.dot(delta_p_global, x_body)
    y_local = np.dot(delta_p_global, y_body)
    z_local = np.dot(delta_p_global, z_body)

    # --- 6. 从机体坐标系下的坐标计算方位角和俯仰角 ---

    # a. 方位角(Azimuth)是X-Y平面上的角度
    # 使用arctan2来正确处理所有象限
    azimuth = np.arctan2(y_local, x_local)

    # b. 俯仰角(Elevation)是目标与X-Y平面的夹角
    # 首先计算目标在X-Y平面上的投影长度
    horizontal_dist = np.sqrt(x_local**2 + y_local**2)
    # 因为我们的Z轴是向下的, 所以z_local为正表示目标在下方, 俯仰角为负
    elevation = np.arctan2(-z_local, horizontal_dist)

    # 返回的azimuth和elevation单位都是弧度(radians)
    return azimuth, elevation, R