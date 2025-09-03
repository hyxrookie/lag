import os
import yaml
import pymap3d
import numpy as np
import math

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


def _calculate_tactical_score(attacker, target, config):
    """
    辅助函数：计算一个单位对另一个单位的战术分数。
    这个函数是计算团队态势优势的基础。建议将其放在一个公共位置（如 utils.py）以便调用。

    Args:
        attacker: 发起攻击或评估优势的智能体对象。
        target: 被评估的目标智能体对象。
        config: 包含所有奖励函数参数的配置对象。

    Returns:
        float: [0, 1] 范围内的综合战术优势分数。
    """
    # --- 从 config 中获取参数，如果不存在则使用默认值 ---
    min_attack_range = getattr(config, 'min_attack_range', 4000.0)
    max_attack_range = getattr(config, 'max_attack_range', 14000.0)
    range_decay_factor = getattr(config, 'range_decay_factor', 0.0005)
    max_ao_rad = math.radians(getattr(config, 'max_missile_attack_angle', 60.0))
    altitude_advantage_ref = getattr(config, 'altitude_advantage_ref', 1000.0)
    velocity_advantage_ref = getattr(config, 'velocity_advantage_ref', 100.0)
    w_geometry = getattr(config, 'w_geometry', 0.6)
    w_energy = getattr(config, 'w_energy', 0.4)
    w_ta_angle = getattr(config, 'w_ta_angle', 0.5)
    w_ao_angle = getattr(config, 'w_ao_angle', 0.3)
    w_range = getattr(config, 'w_range_geom', 0.2)
    w_altitude = getattr(config, 'w_altitude', 0.5)
    w_velocity = getattr(config, 'w_velocity', 0.5)

    # --- 获取运动学特征 ---
    attacker_feature = np.hstack([attacker.get_position(), attacker.get_velocity()])
    target_feature = np.hstack([target.get_position(), target.get_velocity()])

    AO, TA, R = get_AO_TA_R(attacker_feature, target_feature)

    # --- 1. 计算几何优势分数 ---
    ta_score = ((math.cos(TA) + 1.0) / 2.0) if abs(TA) <= max_ao_rad else 0.0
    ao_score = (1.0 - (abs(AO) / max_ao_rad)) if abs(AO) <= max_ao_rad else 0.0
    range_score = 1.0 if min_attack_range <= R <= max_attack_range else 0.0

    geometric_score = w_ta_angle * ta_score + w_ao_angle * ao_score + w_range * range_score

    # --- 2. 计算能量优势分数 ---
    alt_diff = attacker.get_position()[2] - target.get_position()[2]
    alt_score = ((math.tanh(alt_diff / altitude_advantage_ref) + 1.0) / 2.0) if alt_diff >= 0 else 0.0
    vel_diff = np.linalg.norm(attacker.get_velocity()) - np.linalg.norm(target.get_velocity())
    vel_score = ((math.tanh(vel_diff / velocity_advantage_ref) + 1.0) / 2.0) if vel_diff >= 0 else 0.0
    energy_score = w_altitude * alt_score + w_velocity * vel_score

    # --- 3. 计算总战术优势分数 ---
    return w_geometry * geometric_score + w_energy * energy_score
