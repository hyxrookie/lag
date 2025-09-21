import collections
import gzip
import json
import logging
import os
import random
import socket
import struct
import subprocess
import time
import gymnasium
from gymnasium.utils import seeding
import numpy as np
from typing import Dict, Any, Tuple, Set, Union

from envs.JSBSim.core.zk.zk_simulatior import Aircraft, Missile
from envs.JSBSim.envs.env_base import BaseEnv


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

        self._zk_sims = {}  # type: Dict[str, Aircraft]
        self._zk_missiles = {}  # type: Dict[str, Missile]
        self.process = None
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

            # 构建动作命令
            action_input[agent.key][agent.uid] = {
                'mode': 0,
                "fcs/aileron-cmd-norm": norm_action[0],
                "fcs/rudder-cmd-norm": norm_action[1],
                "fcs/elevator-cmd-norm": norm_action[2],
                "fcs/throttle-cmd-norm": norm_action[3],
                "fcs/weapon-launch": norm_action[4],
                # "switch-missile": random.randint(0, 1),
                "change-target": 8,
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
                if current_target_sim is not None and current_target_sim in self.agents:
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

            for i, char in enumerate(target_view_str[::-1]):
                if char == '1':
                    enemy_name = f"{enemy_team_name}_{i}"
                    if enemy_name in self.agents:
                        enemy_aircraft = self.agents[enemy_name]
                        # 将探测到的敌机加入自己阵营的探测集合中
                        temp_detected_sets[aircraft.key].add(enemy_aircraft)
        # 将计算结果（集合）转换为最终的共享列表，存储在局部字典中
        team_detected_lists = {
            "red": sorted(list(temp_detected_sets["red"]), key=lambda x: x.uid),
            "blue": sorted(list(temp_detected_sets["blue"]), key=lambda x: x.uid)
        }
        # --- 步骤 3: 为每架飞机链接到团队共享的探测列表 ---
        for aircraft in self._zk_sims.values():
            # 直接将属性指向局部字典中对应的共享列表
            aircraft.detected_enemies = team_detected_lists[aircraft.key]

    @staticmethod
    def get_common_init_pos():
        max_range = 0.6
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
        red_v, blue_v = 600 * r3, 600 * r4
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