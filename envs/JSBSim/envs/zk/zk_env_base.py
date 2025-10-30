import collections
import gzip
import json
import logging
import math
import os
import random
import socket
import struct
import subprocess
import time
import gymnasium
from gymnasium.utils import seeding
import numpy as np
from typing import Dict, Any, Tuple, Set, Union, Optional, List

from envs.JSBSim.core.zk.zk_simulatior import Aircraft, Missile
from envs.JSBSim.envs.env_base import BaseEnv
from envs.JSBSim.utils.utils import get_AO_TA_R


class ZKBaseEnv(BaseEnv):

    def __init__(self, config_name: str, port):

        super().__init__(config_name)

        self.project_name = getattr(self.config, 'project_name', 'project2')
        self.excute_path = getattr(self.config, 'excute_path', "D:/ZK_20250815/Windows/ZK.exe")
        self.IP = getattr(self.config, 'ip', '127.0.0.1')
        self.PORT = getattr(self.config, 'port', 13000) + port
        self.RENDER = getattr(self.config, 'render', 1)
        self.red_num = 1 if self.project_name == 'project1' else 4
        self.blue_num = 1 if self.project_name == 'project1' else 4
        self.INITIAL = False
        is_success = False
        self.last_send = {}
        self._last_shoot_time = None
        self.min_attack_interval = getattr(self.config, 'min_attack_interval', 125)

        self._zk_sims = {}  # type: Dict[str, Aircraft]
        self._zk_missiles = {}  # type: Dict[str, Missile]
        self.process = None
        self.ruleBasedController = RuleBasedController()
        while not is_success:
            try:
                # 这一部分代码完全不用修改，因为我们使用了参数列表
                # 这是最健壮、最跨平台的方式
                args = [
                    self.excute_path,
                    f'Ip={self.IP}',
                    f'Port={self.PORT}',
                    f'PlayMode={self.RENDER}',
                    f'RedNum={self.red_num}',
                    f'BlueNum={self.blue_num}',
                    'Red=0',
                    'Blue=0',
                    'Scenes=4'
                ]
                print('Creating Env on port {}...'.format(self.PORT))
                print('Executing command:', ' '.join(args))  # 打印命令方便调试

                self.process = subprocess.Popen(args)

                time.sleep(40)
                self._connect()
                is_success = True
                print('Env Created Successfully on port {}'.format(self.PORT))

            except FileNotFoundError:
                # <<< [推荐增加] 专门处理找不到文件的情况
                print(f"错误: 找不到可执行文件 '{self.excute_path}'。请检查路径是否正确以及文件是否存在。")
                # 找不到文件就没必要重试了，直接退出
                raise

            except Exception as e:
                print('Port {} failed to create env: {}'.format(self.PORT, e))
                if self.process:
                    print(f'Terminating failed process (PID: {self.process.pid})...')
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print(f'Process {self.process.pid} did not terminate in time, killing it.')
                        self.process.kill()  # 如果 terminate 不行，就强制 kill

                self.PORT += 50
                time.sleep(5)

    @property
    def agents(self) -> Dict[str, Aircraft]:
        return self._zk_sims

    @property
    def missiles(self) -> Dict[str, Missile]:
        return self._zk_missiles

    def _connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(50)
        print(f'Connecting {self.IP}:{self.PORT}')
        self.socket.connect((self.IP, self.PORT))

    def reconstruct(self):
        print('Reconstruct Env')
        self.create_entity()
        self._connect()
        self.INITIAL = False

    def kill_env(self):
        print('Kill Env')
        output = os.popen(f'netstat -ano | findstr {self.IP}:{self.PORT}')
        output = output.read()
        output = output.split("\n")
        pid = None
        for out_tmp in output:
            out = out_tmp.split(' ')
            out_msg = []
            for msg_tmp in out:
                if msg_tmp != '':
                    out_msg.append(msg_tmp)
            try:
                if out_msg[1] == f'{self.IP}:{self.PORT}':
                    pid = out_msg[-1]
                    break
            except Exception as e:
                print('out_msg', out_msg)
        if pid is not None:
            os.system('taskkill /f /im %s' % pid)
            os.system('kill -9 %s' % pid)
        self.socket.shutdown(socket.SHUT_RDWR)
        self.socket.close()
        self.INITIAL = False

    def _send_condition(self, data):
        self.last_send = data
        # data = json.dumps(data)
        # self.socket[side].send(bytes(data.encode('utf-8')))
        json_str = json.dumps(data)
        compressed = gzip.compress(json_str.encode('utf-8'))
        header = struct.pack('!I', len(compressed))
        self.socket.sendall(header)
        self.socket.sendall(compressed)

    def _recv_all(self, n: int):
        data = bytearray()
        while len(data) < n:
            try:
                remaining = n - len(data)
                packet = self.socket.recv(remaining)
                if not packet:
                    return None
                data.extend(packet)
            except socket.timeout:
                print("接收超时")
                return None
        return bytes(data)

    def _accept_from_socket(self):
        msg_receive = None
        try:
            header = self._recv_all(4)
            if not header:
                return None
            data_len = struct.unpack('!I', header)[0]
            compressed_data = self._recv_all(data_len)
            if not compressed_data:
                return None
            msg_receive = gzip.decompress(compressed_data).decode('utf-8')
            return json.loads(msg_receive)
        except Exception as e:
            print('last send', self.last_send)
            print(e)
            return None
    def postprocess_action(self, action_dict):
        """处理多智能体动作"""
        action_input = {'red': {}, 'blue': {}}

        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            norm_action = self.task.normalize_action(self, agent_id, action)
            rule_decide = self.ruleBasedController.decide(self, agent)
            shoot_interval = self.current_step - self._last_shoot_time[agent_id]
            shoot_flag = shoot_interval > self.min_attack_interval and norm_action[4] == 1 and \
                         rule_decide['fcs/weapon-launch'] == 1
            if shoot_flag:
                self._last_shoot_time[agent_id] = self.current_step
            # 构建动作命令
            action_input[agent.key][agent.uid] = {
                'mode': 0,
                "fcs/aileron-cmd-norm": norm_action[0],
                "fcs/elevator-cmd-norm": norm_action[1],
                "fcs/rudder-cmd-norm": norm_action[2],
                "fcs/throttle-cmd-norm": norm_action[3],
                "fcs/weapon-launch": 1 if shoot_flag else 0,
                "switch-missile": rule_decide['switch-missile'],
                "switch-acmType ": rule_decide['switch-acmType'],
                "change-target": rule_decide['change-target'],
            }

        return action_input

    @staticmethod
    def _decode_entity_code(code_val) -> Union[str, None]:
        """
        【新增】解码 Owner/Target 编码。
        将数字 (e.g., 2.0, 13.0) 转换为飞机名称 (e.g., "blue_0", "blue_3")。
        - 十位代表阵营 (0: red, 1: blue)
        - 个位代表编号
        - 9 或 99 代表无目标
        """
        if code_val is None or int(code_val) in [9, 99]:
            return None

        code = int(code_val)
        team_digit = code // 10
        num_digit = code % 10

        team_name = "red" if team_digit == 0 else "blue"
        return f"{team_name}_{num_digit}"

    def update_from_obs(self, observation_data: dict):
        """
        使用包含飞机和导弹的观测字典来更新所有模拟器状态。
        适配导弹数据在 "missile" 键下的新结构。
        """
        current_missile_keys: Set[str] = set()

        # 遍历观测数据的所有顶层键 ("red", "blue", "missile")
        for key, data in observation_data.items():

            # --- 情况1: 数据是飞机阵营信息 ---
            if key in ["red", "blue"]:
                team_data = data
                if not isinstance(team_data, dict): continue
                for aircraft_name, unit_obs in team_data.items():
                    if aircraft_name not in self._zk_sims:
                        self._zk_sims[aircraft_name] = Aircraft(key=key, uid=aircraft_name)
                    self._zk_sims[aircraft_name].update(unit_obs)

        current_missile_keys: Set[str] = set()
        missile_container = observation_data.get("missile", {})

        if isinstance(missile_container, dict):
            for missile_name, missile_obs in missile_container.items():
                current_missile_keys.add(missile_name)

                missile_instance = self._zk_missiles.get(missile_name)

                # --- A. 如果是新导弹，使用工厂模式创建 ---
                if not missile_instance:
                    parent_name = self._decode_entity_code(missile_obs.get("Owner"))
                    parent_sim = self.agents[parent_name]
                    if parent_sim:
                        try:
                            m_type, m_num_str = missile_name.split('_')
                            target_name = self._decode_entity_code(missile_obs.get("Target"))
                            target_sim = None
                            if target_name is not None and target_name in self.agents:
                                target_sim = self.agents[target_name]

                            missile_instance = Missile.create(
                                missile_type=m_type,
                                number=int(m_num_str),
                                uid=missile_name,
                                parent=parent_sim,
                                target=target_sim
                            )
                            self._zk_missiles[missile_name] = missile_instance
                        except (ValueError, IndexError):
                            continue
                    else:
                        continue

                # --- B. 对所有存在的导弹（包括刚创建的），更新数据和目标 ---
                # 1. 更新导弹自身的基础数据 (速度, 高度等)
                missile_instance.update(missile_obs)

                # 2. 解码并查找当前帧的目标实体
                current_target_name = self._decode_entity_code(missile_obs.get("Target"))
                current_target_sim = None
                if current_target_name is not None:
                    current_target_sim = self.agents[current_target_name]

                # 3. 调用增强的 set_target 方法，它会自动处理目标是否变化
                missile_instance.set_target(current_target_sim)

        # --- 清理已消失的导弹 (逻辑保持不变) ---
        disappeared_missiles = set(self._zk_missiles.keys()) - current_missile_keys
        for missile_name in disappeared_missiles:
            missile_to_remove = self._zk_missiles.get(missile_name)
            if not missile_to_remove: continue

            # 从父飞机的发射列表中移除
            if missile_to_remove.parent and missile_to_remove in missile_to_remove.parent.launched_missiles:
                missile_to_remove.parent.launched_missiles.remove(missile_to_remove)

            # 从目标飞机的被攻击列表中移除
            if missile_to_remove.target and missile_to_remove in missile_to_remove.target.under_missiles:
                missile_to_remove.target.under_missiles.remove(missile_to_remove)

            del self._zk_missiles[missile_name]

        # --- 更新飞机间的连接关系 (逻辑保持不变) ---
        self._update_relationships()

        self._update_team_rosters()

    def _update_team_rosters(self):
        """
        更新 self.red_ids 和 self.blue_ids 列表，包含当前所有飞机的ID。
        """
        # 在更新前清空列表
        self.ego_ids.clear()
        self.enm_ids.clear()

        for aircraft_name, aircraft_sim in self._zk_sims.items():
            if aircraft_sim.key == "red":
                self.ego_ids.append(aircraft_name)
            elif aircraft_sim.key == "blue":
                self.enm_ids.append(aircraft_name)

    def _update_relationships(self):
        """
        更新所有飞机之间的友机(partners)和敌机(enemies)连接关系。
        """
        # 遍历管理器中的每一架飞机
        for subject_aircraft in self._zk_sims.values():
            # 在每次更新前，清空旧的连接关系列表
            subject_aircraft.partners.clear()
            subject_aircraft.enemies.clear()

            # --- 1. 建立完全的友机列表 (Partners) ---
            for other_aircraft in self._zk_sims.values():
                # 如果是自己，则跳过
                if other_aircraft is subject_aircraft:
                    continue
                # 如果阵营相同，则是友机
                if other_aircraft.key == subject_aircraft.key:
                    subject_aircraft.partners.append(other_aircraft)
                else:
                    # 阵营不同，就是敌机
                    subject_aircraft.enemies.append(other_aircraft)

        # --- 步骤 2: 计算每个团队探测到的敌军并集 ---
        # 使用集合来自动去重，实现并集功能
        temp_detected_sets = collections.defaultdict(set)

        for aircraft in self._zk_sims.values():
            target_view_str = str(int(aircraft.get("TargetIntoView")) or "0")
            enemy_team_name = "blue" if aircraft.key == "red" else "red"
            aircraft.single_detected_enemies.clear()
            for i, char in enumerate(target_view_str[::-1]):
                if char == '1':
                    enemy_name = f"{enemy_team_name}_{i}"
                    if enemy_name in self.agents:
                        enemy_aircraft = self.agents[enemy_name]
                        # 将探测到的敌机加入自己阵营的探测集合中
                        temp_detected_sets[aircraft.key].add(enemy_aircraft)
                        aircraft.single_detected_enemies.append(enemy_aircraft)
        # 将计算结果（集合）转换为最终的共享列表，存储在局部字典中
        team_detected_lists = {
            "red": sorted(list(temp_detected_sets["red"]), key=lambda x: x.uid),
            "blue": sorted(list(temp_detected_sets["blue"]), key=lambda x: x.uid)
        }
        # --- 步骤 3: 为每架飞机链接到团队共享的探测列表 ---
        for aircraft in self._zk_sims.values():
            # 直接将属性指向局部字典中对应的共享列表
            aircraft.share_detected_enemies = team_detected_lists[aircraft.key]

    @staticmethod
    def get_common_init_pos():
        max_range = 0.22
        red_y = 0.5 * np.random.random() - 0.25
        blue_y = 0.5 * np.random.random() - 0.25
        initial_pos_set = {
            'equal': [[-max_range, max_range, 90, -90]],
        }
        # 可以设置权重
        key_ = random.choice(list(initial_pos_set.keys()))
        initial_pos = random.choice(initial_pos_set[key_])
        r1 = 0.25 * np.random.random() + 0.75
        r2 = 0.25 * np.random.random() + 0.75
        r3 = 0.25 * np.random.random() + 0.75
        r4 = 0.25 * np.random.random() + 0.75
        r5 = 0.25 * np.random.random() + 0.75
        red_x, blue_x, red_psi, blue_psi = \
            r1 * initial_pos[0], \
            r2 * initial_pos[1], \
                initial_pos[2], \
                initial_pos[3]
        red_v, blue_v = 1200 * r3, 1200 * r4
        h = 32000 * r5
        return red_x, red_y, red_psi, red_v, blue_x, blue_y, blue_psi, blue_v, h

    @staticmethod
    def reset_variable(red_x, red_y, red_psi, red_v, blue_x, blue_y, blue_psi, blue_v, h, red_num, blue_num):
        # 如果有多个飞机，那么就采用近似平行占位，加入一些随机的位置差距
        reset_attribute = {
            'red': {
                'red_0': {
                    "ic/h-sl-ft": h, "ic/terrain-elevation-ft": 1e-08,
                    "ic/long-gc-deg": red_x, "ic/lat-geod-deg": red_y,
                    "ic/u-fps": red_v, "ic/v-fps": 0, "ic/w-fps": 0,
                    "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                    "ic/phi-deg": 0, "ic/theta-deg": 0,
                    "ic/roc-fpm": 0, "ic/psi-true-deg": red_psi}
            },
            'blue': {
                'blue_0': {
                    "ic/h-sl-ft": h, "ic/terrain-elevation-ft": 1e-08,
                    "ic/long-gc-deg": blue_x, "ic/lat-geod-deg": blue_y,
                    "ic/u-fps": blue_v, "ic/v-fps": 0, "ic/w-fps": 0,
                    "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                    "ic/phi-deg": 0, "ic/theta-deg": 0,
                    "ic/roc-fpm": 0, "ic/psi-true-deg": blue_psi}
            }}
        for i in range(red_num - 1):
            reset_attribute['red'][f'red_{i + 1}'] = \
                {
                    "ic/h-sl-ft": h, "ic/terrain-elevation-ft": 1e-08,
                    "ic/long-gc-deg": red_x, "ic/lat-geod-deg": red_y + 0.05 * (i + 1),
                    "ic/u-fps": red_v, "ic/v-fps": 0, "ic/w-fps": 0,
                    "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                    "ic/phi-deg": 0, "ic/theta-deg": 0,
                    "ic/roc-fpm": 0, "ic/psi-true-deg": red_psi
                }
        for i in range(blue_num - 1):
            reset_attribute['blue'][f'blue_{i + 1}'] = \
                {
                    "ic/h-sl-ft": h, "ic/terrain-elevation-ft": 1e-08,
                    "ic/long-gc-deg": blue_x, "ic/lat-geod-deg": blue_y + 0.05 * (i + 1),
                    "ic/u-fps": blue_v, "ic/v-fps": 0, "ic/w-fps": 0,
                    "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                    "ic/phi-deg": 0, "ic/theta-deg": 0,
                    "ic/roc-fpm": 0, "ic/psi-true-deg": blue_psi
                }
        return reset_attribute



    @staticmethod
    # 您可以将这个函数放在您的类中，并根据需要添加 self 或 @staticmethod
    # class AircraftInitialPositionGenerator:
    #     def generate_initial_conditions_dict(self, red_num: int, blue_num: int, min_sep_km: float = 0.5):
    def generate_initial_conditions_dict(red_num: int, blue_num: int, min_sep_km: float = 0.5):
        """
        最终版本：
        - 完全采用先进的基地-集群生成逻辑来确定每架飞机的位置。
        - 完全采用先进逻辑中的方式为每架飞机独立随机化高度、速度和航向。
        - 唯一的参数修改是将基地间距设为 10-60 公里。
        - 最终输出为您指定的字典格式。
        """
        # --- 1. 常量和单位换算 (源自第二段代码) ---
        KM_PER_DEG_LAT = 111.132
        KM_PER_DEG_LON_AT_EQ = 111.320
        FT_PER_METER = 3.28084

        # --- 2. 基地与范围设置 (源自第二段代码) ---
        red_base_lon_deg = 0.0
        red_base_lat_deg = 0.0
        inner_radius_km = 5.0  # 飞机在基地周围 5km 半径内生成

        # --- 这里是您要求的核心修改点 ---
        min_base_separation_km = 20.0
        max_base_separation_km = 50.0

        # --- 3. 蓝队基地位置计算 (源自第二段代码) ---
        km_per_deg_lon_red = KM_PER_DEG_LON_AT_EQ * math.cos(math.radians(red_base_lat_deg))

        angle_rad = random.uniform(0, 2 * math.pi)
        distance_km = random.uniform(min_base_separation_km, max_base_separation_km)
        delta_lat_deg = (distance_km * math.cos(angle_rad)) / KM_PER_DEG_LAT
        delta_lon_deg = (distance_km * math.sin(angle_rad)) / km_per_deg_lon_red
        blue_base_lat_deg = red_base_lat_deg + delta_lat_deg
        blue_base_lon_deg = red_base_lon_deg + delta_lon_deg

        km_per_deg_lon_blue = KM_PER_DEG_LON_AT_EQ * math.cos(math.radians(blue_base_lat_deg))

        # --- 4. 飞机在基地周围的放置算法 (源自第二段代码) ---
        def sample_offset_deg(R_km, km_per_deg_lon):
            u, r = random.random(), R_km * math.sqrt(random.random())
            theta = random.uniform(0.0, 2.0 * math.pi)
            return (r * math.cos(theta)) / KM_PER_DEG_LAT, (r * math.sin(theta)) / km_per_deg_lon

        def far_enough(new_lat, new_lon, placed, km_per_deg_lon, min_sep):
            for lat, lon in placed:
                dlat_km, dlon_km = (new_lat - lat) * KM_PER_DEG_LAT, (new_lon - lon) * km_per_deg_lon
                if (dlat_km ** 2 + dlon_km ** 2) < (min_sep ** 2): return False
            return True

        def place_team(num_aircraft, base_lat, base_lon, km_per_deg_lon, R_km, min_sep):
            placed = []
            for _ in range(num_aircraft):
                attempts, max_attempts, cur_min_sep = 0, 2000, min_sep
                while True:
                    attempts += 1
                    off_lat, off_lon = sample_offset_deg(R_km, km_per_deg_lon)
                    cand_lat, cand_lon = base_lat + off_lat, base_lon + off_lon
                    if far_enough(cand_lat, cand_lon, placed, km_per_deg_lon, cur_min_sep):
                        placed.append((cand_lat, cand_lon));
                        break
                    if attempts >= max_attempts: cur_min_sep *= 0.9; attempts = 0
            return placed

        red_positions = place_team(red_num, red_base_lat_deg, red_base_lon_deg, km_per_deg_lon_red, inner_radius_km,
                                   min_sep_km)
        blue_positions = place_team(blue_num, blue_base_lat_deg, blue_base_lon_deg, km_per_deg_lon_blue,
                                    inner_radius_km, min_sep_km)

        # --- 5. 构建最终的字典 (输出格式源自第一段代码) ---
        reset_attribute = {'red': {}, 'blue': {}}

        # 填充红队数据
        for i in range(red_num):
            # 参数生成完全采用第二段代码的逻辑
            altitude_m = random.randint(5000, 10000)
            heading_deg = random.randint(0, 359)
            speed_mps = random.randint(175, 350)  # 红队速度范围
            lat, lon = red_positions[i]

            aircraft_id = f'red_{i}'
            reset_attribute['red'][aircraft_id] = {
                "ic/h-sl-ft": altitude_m * FT_PER_METER,
                "ic/terrain-elevation-ft": 1e-08,
                "ic/long-gc-deg": lon,
                "ic/lat-geod-deg": lat,
                "ic/u-fps": speed_mps * FT_PER_METER,
                "ic/psi-true-deg": heading_deg,
                # 补全其他参数以确保初始状态稳定
                "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0,
                "ic/q-rad_sec": 0, "ic/r-rad_sec": 0, "ic/phi-deg": 0,
                "ic/theta-deg": 0, "ic/roc-fpm": 0,
            }

        # 填充蓝队数据
        for i in range(blue_num):
            # 参数生成完全采用第二段代码的逻辑
            altitude_m = random.randint(5000, 10000)
            heading_deg = random.randint(0, 359)
            speed_mps = random.randint(175, 350)  # 蓝队速度范围
            lat, lon = blue_positions[i]

            aircraft_id = f'blue_{i}'
            reset_attribute['blue'][aircraft_id] = {
                "ic/h-sl-ft": altitude_m * FT_PER_METER,
                "ic/terrain-elevation-ft": 1e-08,
                "ic/long-gc-deg": lon,
                "ic/lat-geod-deg": lat,
                "ic/u-fps": speed_mps * FT_PER_METER,
                "ic/psi-true-deg": heading_deg,
                # 补全其他参数
                "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0,
                "ic/q-rad_sec": 0, "ic/r-rad_sec": 0, "ic/phi-deg": 0,
                "ic/theta-deg": 0, "ic/roc-fpm": 0,
            }

        return reset_attribute

    def render(self, mode="txt", filepath='./JSBSimRecording.txt.acmi', tacview=None):
        """Renders the environment.

        The set of supported modes varies per environment. (And some

        environments do not support rendering at all.) By convention,

        if mode is:

        - human: print on the terminal
        - txt: output to txt.acmi files
        - real_time: realtime render with tacview by socket comm

        Note:

            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        :param mode: str, the mode to render with
        """
        if mode == "txt":
            if not self._create_records:
                with open(filepath, mode='w', encoding='utf-8-sig') as f:
                    f.write("FileType=text/acmi/tacview\n")
                    f.write("FileVersion=2.1\n")
                    f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")
                self._create_records = True
            with open(filepath, mode='a', encoding='utf-8-sig') as f:
                timestamp = self.current_step * self.time_interval
                f.write(f"#{timestamp:.2f}\n")
                for sim in self._zk_sims.values():
                    log_msg = sim.log()
                    if log_msg is not None:
                        f.write(log_msg + "\n")
                for sim in self._zk_missiles.values():
                    log_msg = sim.log()
                    if log_msg is not None:
                        f.write(log_msg + "\n")
        elif mode == "real_time":
            timestamp = self.current_step * self.time_interval
            data = [f"#{timestamp:.2f}\n"]
            for sim in self._zk_sims.values():
                log_msg = sim.log()
                if log_msg is not None:
                    data.append(log_msg + "\n")

            for sim in self._zk_missiles.values():
                log_msg = sim.log()
                if log_msg is not None:
                    data.append(log_msg + "\n")

            data_str = "".join(data)
            # send data to tacview
            tacview.send_data_to_client(data_str)
        else:
            raise NotImplementedError


class RuleBasedController:
    """
    一个无状态的、反应式的无人机对抗规则控制器。
    它在每个时间步都根据当前的战场“事实”做出最优决策，
    代码简洁且健壮。
    """

    def __init__(self, sraam_max_range_m=18000.0, crm_preferred_min_range_m=20000.0, max_missiles_per_target=2):
        self.SRAAM_MAX_RANGE_M = sraam_max_range_m
        self.CRM_PREFERRED_MIN_RANGE_M = crm_preferred_min_range_m
        self.MAX_MISSILES_PER_TARGET = max_missiles_per_target

    # --- 辅助方法 (无需改动) ---

    def _parse_id_from_name(self, name: str) -> Optional[int]:
        try:
            return int(name.split('_')[-1])
        except (ValueError, IndexError):
            return None

    def _parse_target_list_string(self, target_str: str) -> List[int]:
        if not target_str or not isinstance(target_str, str): return []
        return [int(tid) for tid in target_str.split('/') if tid.isdigit()]

    def _find_priority_target(self, agent: Aircraft) -> Tuple[Optional[Aircraft], float]:

        priority_target = None
        target_distance = 210000.0
        agent_feature = np.hstack([agent.get_position(), agent.get_velocity()])
        for enm in agent.single_detected_enemies:
            if enm.is_alive:
                enm_feature = np.hstack([enm.get_position(), enm.get_velocity()])
                _, _, r = get_AO_TA_R(agent_feature, enm_feature)
                if r < target_distance:
                    priority_target = enm
                    target_distance = r

        return priority_target, target_distance

    def _get_sorted_targets(self, agent: Aircraft) -> List[Aircraft]:
        """
        获取所有探测到的、存活的敌机，并按距离从近到远排序。
        """
        target_candidates = []
        agent_feature = np.hstack([agent.get_position(), agent.get_velocity()])

        for enm in agent.single_detected_enemies:
            if enm.is_alive:
                enm_feature = np.hstack([enm.get_position(), enm.get_velocity()])
                _, _, r = get_AO_TA_R(agent_feature, enm_feature)
                target_candidates.append((r, enm))

        target_candidates.sort(key=lambda x: x[0])
        return [enm for r, enm in target_candidates]

    def _get_missiles_targeting_enemies(self, env: ZKBaseEnv, agent_team: str) -> Dict[int, int]:
        """
        统计每个敌机当前被多少枚我方在途导弹攻击。
        """
        targeting_info = {}
        for missile in env.missiles.values():
            if missile.is_alive and missile.parent and agent_team in missile.parent.key and missile.target:
                target_id = missile.target.uid
                targeting_info[target_id] = targeting_info.get(target_id, 0) + 1
        return targeting_info

    # --- 主决策方法 (无状态逻辑) ---

    def decide(self, env: ZKBaseEnv, agent: Aircraft) -> Dict:
        """为给定的Agent状态生成一套无状态的、反应式的规则化指令。"""
        # --- 初始化 ---
        commands = {
            'fcs/weapon-launch': 0, 'switch-missile': 0,
            'switch-acmType': 0, 'change-target': 9
        }

        # --- 1. 全局态势感知 ---
        sorted_targets = self._get_sorted_targets(agent)
        if agent.get('MissileAlert') == 1 or not sorted_targets:
            return commands

        agent_team = 'red' if 'red' in agent.key else 'blue'
        missiles_on_targets = self._get_missiles_targeting_enemies(env, agent_team)

        # --- 2. 确定本回合的“有效战术目标” ---
        effective_target = None
        for target in sorted_targets:
            num_missiles_on_this_target = missiles_on_targets.get(target.uid, 0)
            if num_missiles_on_this_target < self.MAX_MISSILES_PER_TARGET:
                effective_target = target
                break

        if effective_target is None:
            return commands

        # --- 3. 武器系统管理 (基于有效目标) ---
        agent_pos = agent.get_position()
        target_pos = effective_target.get_position()
        target_distance = np.linalg.norm(agent_pos - target_pos)
        AimMode = agent.get("AimMode")

        if target_distance < self.SRAAM_MAX_RANGE_M and agent.get('SRAAMCurrentNum') > 0 and AimMode == 1:
            commands['switch-missile'] = 1
        elif target_distance > self.CRM_PREFERRED_MIN_RANGE_M and agent.get('AMRAAMCurrentNum') > 0 and AimMode == 0:
            commands['switch-missile'] = 1

        if AimMode == 0:
            alt_diff = abs(agent.get('position/h-sl-ft') - effective_target.get('position/h-sl-ft'))
            if alt_diff > 5000 and agent.get("ACMaimMode") == 0:
                commands['switch-acmType'] = 1
            elif alt_diff < 4500 and agent.get("ACMaimMode") == 1:
                commands['switch-acmType'] = 1

        if commands['switch-missile'] == 1 or commands['switch-acmType'] == 1:
            return commands

        # --- 4. 核心决策：锁定与发射 (围绕有效目标展开) ---
        locked_id = None
        if AimMode == 0:
            locked_id = self.get_priority_sraam_target(agent, agent.get('SRAAMTargetLocked'))
        elif AimMode == 1:
            locked_id = self.get_priority_amraam_target(agent, agent.get('AMRAAMlockedTarget'))

        if locked_id == effective_target.uid:
            # 分支 A: 目标正确，执行“发射”流程
            locked_agent = env.agents[locked_id]
            if not locked_agent:
                return commands

            agent_feature = np.hstack([agent.get_position(), agent.get_velocity()])
            locked_agent_feature = np.hstack([locked_agent.get_position(), locked_agent.get_velocity()])
            _, _, r = get_AO_TA_R(agent_feature, locked_agent_feature)
            envelope_min = agent.get('EnvelopeMin')
            envelope_max = agent.get('EnvelopeMax')
            is_in_envelope = envelope_min <= r <= envelope_max

            if is_in_envelope:
                commands['fcs/weapon-launch'] = 1
        else:
            # 分支 B: 目标错误(或无目标)，执行“切换锁定”流程
            change_target_cmd = self.generate_lock_command([effective_target.uid])
            commands['change-target'] = change_target_cmd

        return commands

    def generate_lock_command(self, target_names: List[str]) -> int:
        """
        根据要锁定的敌方飞机名称列表，生成对应的整数指令。

        该函数基于10的幂次规则：
        - 'blue_0' -> 10^0 = 1
        - 'blue_1' -> 10^1 = 10
        - 'blue_2' -> 10^2 = 100
        最终指令是所有目标对应值的总和。

        参数:
            target_names (List[str]): 一个包含敌机名称字符串的列表。
                                      例如: ['blue_0', 'blue_3']

        返回:
            int: 计算出的最终锁定指令。如果列表为空，返回0。
        """
        command = 0
        if not target_names:
            return 0

        for name in target_names:
            try:
                # 从 'blue_3' 这样的字符串中分离出数字 '3'
                parts = name.split('_')
                if len(parts) > 1 and parts[-1].isdigit():
                    enemy_index = int(parts[-1])
                    # 计算 10 的幂并加到总指令中
                    command += 10 ** enemy_index
                else:
                    print(f"警告: 无法从 '{name}' 中解析敌机编号，已跳过。")
            except ValueError:
                print(f"警告: '{name}' 中的编号部分不是有效数字，已跳过。")
            except Exception as e:
                print(f"处理 '{name}' 时发生未知错误: {e}，已跳过。")

        return command

    def get_priority_amraam_target(self, agent: Aircraft,
                                   locked_target_value: Union[float, int, None],
                                   ) -> Optional[str]:
        """
        解析 AMRAAMlockedTarget 的编码值，并根据新规则返回最高优先级的目标全名。

        新优先规则:
        - 从左到右，遇到的第一个非'9'的数字即为最高优先级目标。
        - 例如，对于值 3199，最高优先级目标是 3，而不是 1。

        新返回格式:
        - 函数根据 agent_name 判断其阵营（'red' 或 'blue'）。
        - 如果 agent 是 'red'，则返回 'blue_X' 格式的名称。
        - 如果 agent 是 'blue'，则返回 'red_X' 格式的名称。

        参数:
            locked_target_value (Union[float, int, None]):
                从平台获取的 AMRAAMlockedTarget 值。
            agent_name (str):
                执行此决策的Agent的名称，例如 "red_0" 或 "blue_1"。

        返回:
            Optional[str]: 最高优先级敌机的完整名称 (如 "blue_3")。
                           如果没有锁定目标或输入无效，则返回 None。
        """
        # --- 步骤 1: 预处理输入 (与之前相同) ---
        locked_value_as_int = 0
        try:
            if locked_target_value is None: return None
            locked_value_as_int = int(locked_target_value)
        except (ValueError, TypeError):
            print(f"警告: 无效的 AMRAAMlockedTarget 输入 '{locked_target_value}'，无法解析。")
            return None

        # --- 步骤 2: 判断敌方阵营 (新逻辑) ---
        target_team_prefix = ""
        if 'red' in agent.key:
            target_team_prefix = "blue"
        elif 'blue' in agent.key:
            target_team_prefix = "red"
        else:
            print(f"警告: 无法从 agent_name '{agent.key}' 判断阵营。")
            return None

        # --- 步骤 3: 核心解析逻辑 (实现新的优先规则) ---

        # 处理哨兵值
        if locked_value_as_int == 9999:
            return None

        # 转换为4位字符串
        target_str = f"{locked_value_as_int:04d}"

        # 遍历字符串，找到第一个非'9'的字符
        for digit_char in target_str:
            if digit_char != '9':
                # 找到了！这就是最高优先级的目标
                priority_target_id = int(digit_char)
                # 构建完整的敌机名称并立即返回
                return f"{target_team_prefix}_{priority_target_id}"

        # 如果循环结束都没有找到 (例如值是9999)，则没有锁定目标
        return None

    def get_priority_sraam_target(self, agent: Aircraft,
                                  locked_target_value: Union[float, int, None],
                                  ) -> Optional[str]:
        """
        解析 SRAAMlockedTarget 的编码值，并根据新规则返回最高优先级的目标全名。

        新优先规则:
        - 如果是9表示没有锁定， 如果不是9就是锁定的id

        新返回格式:
        - 函数根据 agent_name 判断其阵营（'red' 或 'blue'）。
        - 如果 agent 是 'red'，则返回 'blue_X' 格式的名称。
        - 如果 agent 是 'blue'，则返回 'red_X' 格式的名称。

        参数:
            locked_target_value (Union[float, int, None]):
                从平台获取的 SRAAMlockedTarget 值。
            agent_name (str):
                执行此决策的Agent的名称，例如 "red_0" 或 "blue_1"。

        返回:
            Optional[str]: 最高优先级敌机的完整名称 (如 "blue_3")。
                           如果没有锁定目标或输入无效，则返回 None。
        """
        # 预处理输入
        try:
            if locked_target_value is None:
                return None
            locked_value_as_int = int(locked_target_value)
        except (ValueError, TypeError):
            print(f"警告: 无效的 SRAAMTargetLocked 输入 '{locked_target_value}'，无法解析。")
            return None

        # 判断敌方阵营
        if 'red' in agent.key:
            target_team_prefix = "blue"
        elif 'blue' in agent.key:
            target_team_prefix = "red"
        else:
            print(f"警告: 无法从 agent_name '{agent.key}' 判断阵营。")
            return None

        # 处理哨兵值
        if locked_value_as_int == 9:
            return None

        # 直接返回锁定的目标ID
        return f"{target_team_prefix}_{locked_value_as_int}"
