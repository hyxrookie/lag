import numpy as np
from typing import Tuple, Dict, Any
from .env_base import BaseEnv
from ..tasks.multiplecombat_task import HierarchicalMultipleCombatShootTask, HierarchicalMultipleCombatTask, MultipleCombatTask
import random
import math

from ..utils.utils import get_AO_TA_R


class MultipleCombatEnv(BaseEnv):
    """
    MultipleCombatEnv is an multi-player competitive environment.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        self._create_records = False
        self.episode_metrics = None

    @property
    def share_observation_space(self):
        return self.task.share_observation_space

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'multiplecombat':
            self.task = MultipleCombatTask(self.config)
        elif taskname == 'hierarchical_multiplecombat':
            self.task = HierarchicalMultipleCombatTask(self.config)
        elif taskname == 'hierarchical_multiplecombat_shoot':
            self.task = HierarchicalMultipleCombatShootTask(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Resets the state of the environment and returns an initial observation.

        Returns:
            obs (dict): {agent_id: initial observation}
            share_obs (dict): {agent_id: initial state}
        """
        self._create_records = False
        self.current_step = 0
        self.reset_simulators()
        self.task.reset(self)
        obs = self.get_obs()
        share_obs = self.get_state()
        self._init_episode_metrics()
        return self._pack(obs), self._pack(share_obs)

    def reset_simulators(self):
        seed = getattr(self, "_reset_seed", None)
        swap_red_blue = getattr(self, "_reset_swap_red_blue", False)
        reset_mode = getattr(self, "_reset_mode", "symmetric")

        if reset_mode == "normal":
            self.my_random_reset_simulators(seed=seed)
        elif reset_mode == "symmetric":
            self.my_random_reset_simulators(
                seed=seed,
                swap_red_blue=swap_red_blue,
                symmetric_pairing=True,
            )
        else:
            raise ValueError(f"Unknown reset_mode: {reset_mode}")

    def set_reset_config(
            self,
            seed=None,
            swap_red_blue=False,
            reset_mode="symmetric",
    ):
        """
        设置下一次 reset 使用的初始化参数。

        Args:
            seed: 当前 episode 的随机种子。
            swap_red_blue: 是否红蓝初始态势互换。
            reset_mode:
                "normal"     普通随机初始化
                "symmetric"  对称随机初始化
        """
        self._reset_seed = seed
        self._reset_swap_red_blue = swap_red_blue
        self._reset_mode = reset_mode
    def my_random_reset_simulators(
            self,
            seed=None,
            swap_red_blue=False,
            symmetric_pairing=True,
            center_lon_deg=120.0,
            center_lat_deg=60.0,
            min_team_separation_km=10.0,
            max_team_separation_km=40.0,
            inner_radius_km=5.0,
            altitude_range_m=(5000, 10000),
            speed_range_mps=(200, 300),
            heading_jitter_deg=20.0,
    ):
        """
        随机但尽量公平的红蓝双方初始化。

        特点：
        1. 红蓝双方关于战场中心近似对称；
        2. 支持 seed，保证可复现；
        3. 支持 swap_red_blue，用于红蓝互换评估；
        4. 支持成对镜像散布，减少初始位置偏置；
        5. 默认双方大致相向飞行，避免一方天然占优。

        Args:
            seed: 随机种子。相同 seed + 相同参数会生成相同初始场景。
            swap_red_blue: 是否交换红蓝双方初始态势。
            symmetric_pairing: 是否按 A1-B1, A2-B2 进行镜像初始化。
            center_lon_deg: 战场中心经度。
            center_lat_deg: 战场中心纬度。
            min_team_separation_km: 红蓝中心点最小间距。
            max_team_separation_km: 红蓝中心点最大间距。
            inner_radius_km: 队伍内部散布半径。
            altitude_range_m: 高度范围，单位 m。
            speed_range_mps: 速度范围，单位 m/s。
            heading_jitter_deg: 初始航向扰动，单位 deg。
        """

        # =========================
        # 1. 常量与局部随机数生成器
        # =========================
        KM_PER_DEG_LAT = 111.132
        KM_PER_DEG_LON_AT_EQ = 111.320
        FT_PER_METER = 3.28084

        rng = random.Random(seed)

        # 当前纬度附近的经度换算
        km_per_deg_lon_center = KM_PER_DEG_LON_AT_EQ * math.cos(math.radians(center_lat_deg))
        if abs(km_per_deg_lon_center) < 1e-6:
            raise ValueError("center_lat_deg too close to poles, longitude conversion becomes unstable.")

        def local_km_to_lonlat(x_km, y_km):
            """
            局部平面坐标转经纬度。
            x_km: 东向偏移，单位 km
            y_km: 北向偏移，单位 km
            """
            lon = center_lon_deg + x_km / km_per_deg_lon_center
            lat = center_lat_deg + y_km / KM_PER_DEG_LAT
            return lon, lat

        def wrap_heading_deg(deg):
            return deg % 360.0

        def heading_from_vector_deg(dx_km, dy_km):
            """
            根据局部平面向量计算 JSBSim 常用航向角：
            0 deg = 北，90 deg = 东。
            dx_km: 东向
            dy_km: 北向
            """
            return wrap_heading_deg(math.degrees(math.atan2(dx_km, dy_km)))

        def sample_uniform_disk(radius_km):
            """
            在圆盘内均匀采样一个点。
            注意不能直接 r uniform，否则会让点集中在中心。
            """
            theta = rng.uniform(0.0, 2.0 * math.pi)
            r = radius_km * math.sqrt(rng.uniform(0.0, 1.0))
            x = r * math.sin(theta)  # 东向
            y = r * math.cos(theta)  # 北向
            return x, y

        # =========================
        # 2. 获取红蓝飞机 id
        # =========================
        red_ids = sorted([sim_id for sim_id in self._jsbsims.keys() if sim_id.startswith("A")])
        blue_ids = sorted([sim_id for sim_id in self._jsbsims.keys() if sim_id.startswith("B")])

        if len(red_ids) == 0 or len(blue_ids) == 0:
            raise ValueError("No red or blue simulators found. Red ids should start with 'A', blue ids with 'B'.")

        # =========================
        # 3. 随机生成红蓝中心点，关于战场中心对称
        # =========================
        # 红蓝中心距离
        team_sep_km = rng.uniform(min_team_separation_km, max_team_separation_km)

        # 红蓝连线方向，局部坐标中 angle=0 表示北向
        line_angle_rad = rng.uniform(0.0, 2.0 * math.pi)

        # 从红方中心指向蓝方中心的单位向量
        ux = math.sin(line_angle_rad)  # 东向
        uy = math.cos(line_angle_rad)  # 北向

        half_sep = team_sep_km / 2.0

        canonical_red_center = (-half_sep * ux, -half_sep * uy)
        canonical_blue_center = (half_sep * ux, half_sep * uy)

        # 红方默认朝向蓝方，蓝方默认朝向红方
        red_base_heading = heading_from_vector_deg(
            canonical_blue_center[0] - canonical_red_center[0],
            canonical_blue_center[1] - canonical_red_center[1],
        )
        blue_base_heading = wrap_heading_deg(red_base_heading + 180.0)

        # =========================
        # 4. 生成 canonical 初始状态
        # =========================
        # canonical 表示还没有做红蓝互换之前的初始状态
        canonical_states = {}

        if symmetric_pairing:
            # 成对镜像初始化：A_i 的内部偏移为 offset，B_i 为 -offset
            pair_num = min(len(red_ids), len(blue_ids))

            for i in range(pair_num):
                red_id = red_ids[i]
                blue_id = blue_ids[i]

                offset_x, offset_y = sample_uniform_disk(inner_radius_km)

                # 高度和速度也可以做近似对称：同一对飞机使用相同基础值
                altitude_m = rng.uniform(*altitude_range_m)
                speed_mps = rng.uniform(*speed_range_mps)

                # 航向扰动镜像：红方 +jitter，蓝方 -jitter
                jitter = rng.uniform(-heading_jitter_deg, heading_jitter_deg)

                red_x = canonical_red_center[0] + offset_x
                red_y = canonical_red_center[1] + offset_y
                blue_x = canonical_blue_center[0] - offset_x
                blue_y = canonical_blue_center[1] - offset_y

                red_lon, red_lat = local_km_to_lonlat(red_x, red_y)
                blue_lon, blue_lat = local_km_to_lonlat(blue_x, blue_y)

                canonical_states[red_id] = {
                    "lon": red_lon,
                    "lat": red_lat,
                    "altitude_m": altitude_m,
                    "heading_deg": wrap_heading_deg(red_base_heading + jitter),
                    "speed_mps": speed_mps,
                }

                canonical_states[blue_id] = {
                    "lon": blue_lon,
                    "lat": blue_lat,
                    "altitude_m": altitude_m,
                    "heading_deg": wrap_heading_deg(blue_base_heading - jitter),
                    "speed_mps": speed_mps,
                }

            # 如果红蓝数量不一致，剩余飞机单独随机生成
            for red_id in red_ids[pair_num:]:
                offset_x, offset_y = sample_uniform_disk(inner_radius_km)
                x = canonical_red_center[0] + offset_x
                y = canonical_red_center[1] + offset_y
                lon, lat = local_km_to_lonlat(x, y)

                canonical_states[red_id] = {
                    "lon": lon,
                    "lat": lat,
                    "altitude_m": rng.uniform(*altitude_range_m),
                    "heading_deg": wrap_heading_deg(
                        red_base_heading + rng.uniform(-heading_jitter_deg, heading_jitter_deg)),
                    "speed_mps": rng.uniform(*speed_range_mps),
                }

            for blue_id in blue_ids[pair_num:]:
                offset_x, offset_y = sample_uniform_disk(inner_radius_km)
                x = canonical_blue_center[0] + offset_x
                y = canonical_blue_center[1] + offset_y
                lon, lat = local_km_to_lonlat(x, y)

                canonical_states[blue_id] = {
                    "lon": lon,
                    "lat": lat,
                    "altitude_m": rng.uniform(*altitude_range_m),
                    "heading_deg": wrap_heading_deg(
                        blue_base_heading + rng.uniform(-heading_jitter_deg, heading_jitter_deg)),
                    "speed_mps": rng.uniform(*speed_range_mps),
                }

        else:
            # 非成对模式：红蓝仍然关于中心对称，但内部散布独立随机
            for red_id in red_ids:
                offset_x, offset_y = sample_uniform_disk(inner_radius_km)
                x = canonical_red_center[0] + offset_x
                y = canonical_red_center[1] + offset_y
                lon, lat = local_km_to_lonlat(x, y)

                canonical_states[red_id] = {
                    "lon": lon,
                    "lat": lat,
                    "altitude_m": rng.uniform(*altitude_range_m),
                    "heading_deg": wrap_heading_deg(
                        red_base_heading + rng.uniform(-heading_jitter_deg, heading_jitter_deg)),
                    "speed_mps": rng.uniform(*speed_range_mps),
                }

            for blue_id in blue_ids:
                offset_x, offset_y = sample_uniform_disk(inner_radius_km)
                x = canonical_blue_center[0] + offset_x
                y = canonical_blue_center[1] + offset_y
                lon, lat = local_km_to_lonlat(x, y)

                canonical_states[blue_id] = {
                    "lon": lon,
                    "lat": lat,
                    "altitude_m": rng.uniform(*altitude_range_m),
                    "heading_deg": wrap_heading_deg(
                        blue_base_heading + rng.uniform(-heading_jitter_deg, heading_jitter_deg)),
                    "speed_mps": rng.uniform(*speed_range_mps),
                }

        # =========================
        # 5. 可选红蓝互换
        # =========================
        # swap_red_blue=True 时：
        # A_i 使用 B_i 的初始位置和航向；
        # B_i 使用 A_i 的初始位置和航向。
        #
        # 这样可以检验模型是否存在红蓝阵营偏置。
        final_states = dict(canonical_states)

        if swap_red_blue:
            pair_num = min(len(red_ids), len(blue_ids))

            for i in range(pair_num):
                red_id = red_ids[i]
                blue_id = blue_ids[i]

                red_state = canonical_states[red_id]
                blue_state = canonical_states[blue_id]

                final_states[red_id] = blue_state
                final_states[blue_id] = red_state

        # =========================
        # 6. 写入 JSBSim 初始条件
        # =========================
        for sim_id, sim in self._jsbsims.items():
            state = final_states[sim_id]

            sim.reload({
                "ic_long_gc_deg": state["lon"],
                "ic_lat_geod_deg": state["lat"],
                "ic_h_sl_ft": state["altitude_m"] * FT_PER_METER,
                "ic_psi_true_deg": state["heading_deg"],
                "ic_u_fps": state["speed_mps"] * FT_PER_METER,
            })

        self._tempsims.clear()
    def random_reset_simulators(self):
        # --- 常量定义 ---
        KM_PER_DEG_LAT = 111.132  # 每度纬度对应的公里数 (近似值)
        KM_PER_DEG_LON_AT_EQ = 111.320  # 赤道上每度经度对应的公里数 (近似值)
        FT_PER_METER = 3.28084

        # --- 基地和距离设置 ---
        red_base_lon_deg = 120.0
        red_base_lat_deg = 60.0
        inner_radius_km = 5.0  # 队伍内部散布半径
        min_base_separation_km = 10.0  # 队伍基地最小间距
        max_base_separation_km = 40.0  # (可选) 队伍基地最大间距，增加随机性

        # --- 计算红队纬度处的经度换算因子 ---
        # 注意：math.cos() 需要弧度
        km_per_deg_lon_red = KM_PER_DEG_LON_AT_EQ * math.cos(math.radians(red_base_lat_deg))

        # --- 计算蓝队基地的随机位置 ---
        # 1. 随机选择一个方向 (角度)
        angle_rad = random.uniform(0, 2 * math.pi)
        # 2. 随机选择一个距离 (大于等于最小间距)
        distance_km = random.uniform(min_base_separation_km, max_base_separation_km)

        # 3. 计算经纬度偏移量 (使用平面近似，对于几十公里通常足够)
        delta_lat_deg = (distance_km * math.cos(angle_rad)) / KM_PER_DEG_LAT
        # 使用红队基地的经度换算因子作为近似
        delta_lon_deg = (distance_km * math.sin(angle_rad)) / km_per_deg_lon_red

        # 4. 计算蓝队基准点
        blue_base_lon_deg = red_base_lon_deg + delta_lon_deg
        blue_base_lat_deg = red_base_lat_deg + delta_lat_deg

        # --- 计算蓝队纬度处的经度换算因子 ---
        km_per_deg_lon_blue = KM_PER_DEG_LON_AT_EQ * math.cos(math.radians(blue_base_lat_deg))

        # --- 计算内部散布的最大经纬度偏移量 ---
        #   (单位: 度)
        max_lat_offset_deg = inner_radius_km / KM_PER_DEG_LAT
        max_lon_offset_deg_red = inner_radius_km / km_per_deg_lon_red
        max_lon_offset_deg_blue = inner_radius_km / km_per_deg_lon_blue

        # --- 循环设置每个单位的初始条件 ---
        for sim_id, sim in self._jsbsims.items():
            # 为每个单位生成独立的随机属性
            altitude_m = random.randint(5000, 10000)  # 先用米，方便理解
            heading_deg = random.randint(0, 359)  # 0-359 更常用
            speed_mps = random.randint(200, 300)  # 先用米/秒

            if sim_id.startswith('A'):  # 红队
                # 在基准点周围随机偏移 (允许负值)
                offset_lat = random.uniform(-max_lat_offset_deg, max_lat_offset_deg)
                offset_lon = random.uniform(-max_lon_offset_deg_red, max_lon_offset_deg_red)

                sim.reload({
                    "ic_long_gc_deg": red_base_lon_deg + offset_lon,
                    "ic_lat_geod_deg": red_base_lat_deg + offset_lat,
                    "ic_h_sl_ft": altitude_m * FT_PER_METER,
                    "ic_psi_true_deg": heading_deg,
                    "ic_u_fps": speed_mps * FT_PER_METER,  # 假设 ic_u_fps 是总速度标量
                })
            elif sim_id.startswith('B'):  # 蓝队
                # 在基准点周围随机偏移 (允许负值)
                offset_lat = random.uniform(-max_lat_offset_deg, max_lat_offset_deg)
                offset_lon = random.uniform(-max_lon_offset_deg_blue, max_lon_offset_deg_blue)

                sim.reload({
                    "ic_long_gc_deg": blue_base_lon_deg + offset_lon,
                    "ic_lat_geod_deg": blue_base_lat_deg + offset_lat,
                    "ic_h_sl_ft": altitude_m * FT_PER_METER,
                    "ic_psi_true_deg": heading_deg,
                    "ic_u_fps": speed_mps * FT_PER_METER,
                })

        self._tempsims.clear()
    def new_random_reset_simulators(self, min_sep_km: float = 0.5):
        import math, random

        # --- 内部辅助函数：根据高度和马赫数计算真速 (m/s) ---
        def get_speed_from_mach(altitude_m, target_mach):
            # 国际标准大气 (ISA) 常量
            GAMMA = 1.4  # 空气绝热指数
            R = 287.05  # 气体常数
            T0 = 288.15  # 海平面标准温度 (K)
            L = 0.0065  # 温度随高度递减率 (K/m)

            # 1. 计算该高度的气温 (Kelvin)
            # 对流层顶 (11000m) 以下使用线性递减，以上暂按恒温处理(简单模型)
            clamped_alt = min(altitude_m, 11000.0)
            temperature = T0 - L * clamped_alt

            # 2. 计算音速 a = sqrt(gamma * R * T)
            speed_of_sound = math.sqrt(GAMMA * R * temperature)

            # 3. 计算真速
            return target_mach * speed_of_sound

        # --- 常量定义 ---
        KM_PER_DEG_LAT = 111.132
        KM_PER_DEG_LON_AT_EQ = 111.320
        FT_PER_METER = 3.28084

        # --- 基地与范围设置 ---
        red_base_lon_deg = 120.0
        red_base_lat_deg = 60.0
        inner_radius_km = 5.0
        min_base_separation_km = 50.0  # 30km
        max_base_separation_km = 100.0  # 50km

        # 红队经度换算系数
        km_per_deg_lon_red = KM_PER_DEG_LON_AT_EQ * math.cos(math.radians(red_base_lat_deg))

        # 1. 随机生成蓝队基地（相对红队）
        angle_rad = random.uniform(0, 2 * math.pi)
        distance_km = random.uniform(min_base_separation_km, max_base_separation_km)

        delta_lat_deg = (distance_km * math.cos(angle_rad)) / KM_PER_DEG_LAT
        delta_lon_deg = (distance_km * math.sin(angle_rad)) / km_per_deg_lon_red

        blue_base_lat_deg = red_base_lat_deg + delta_lat_deg
        blue_base_lon_deg = red_base_lon_deg + delta_lon_deg

        # 蓝队经度换算系数
        km_per_deg_lon_blue = KM_PER_DEG_LON_AT_EQ * math.cos(math.radians(blue_base_lat_deg))

        # --- 计算理想交战航向 ---
        red_ideal_heading = math.degrees(angle_rad) % 360
        blue_ideal_heading = (red_ideal_heading + 180) % 360

        # --- 工具函数定义 (位置生成) ---
        def sample_offset_deg(R_km: float, km_per_deg_lon: float):
            u = random.random()
            r = R_km * math.sqrt(u)
            theta = random.uniform(0.0, 2.0 * math.pi)
            dlat_km = r * math.cos(theta)
            dlon_km = r * math.sin(theta)
            return dlat_km / KM_PER_DEG_LAT, dlon_km / km_per_deg_lon

        def far_enough(new_lat_deg, new_lon_deg, placed_list, km_per_deg_lon: float, min_sep: float):
            for (lat_deg, lon_deg) in placed_list:
                dlat_km = (new_lat_deg - lat_deg) * KM_PER_DEG_LAT
                dlon_km = (new_lon_deg - lon_deg) * km_per_deg_lon
                if (dlat_km * dlat_km + dlon_km * dlon_km) < (min_sep * min_sep):
                    return False
            return True

        # --- 分组与位置生成 ---
        red_ids = [sid for sid in self._jsbsims.keys() if sid.startswith('A')]
        blue_ids = [sid for sid in self._jsbsims.keys() if sid.startswith('B')]
        other_ids = [sid for sid in self._jsbsims.keys() if not (sid.startswith('A') or sid.startswith('B'))]

        def place_team(team_ids, base_lat_deg, base_lon_deg, km_per_deg_lon, R_km, min_sep):
            placed = []
            for _ in team_ids:
                attempts = 0
                cur_min_sep = min_sep
                while True:
                    attempts += 1
                    off_lat_deg, off_lon_deg = sample_offset_deg(R_km, km_per_deg_lon)
                    cand_lat = base_lat_deg + off_lat_deg
                    cand_lon = base_lon_deg + off_lon_deg
                    if far_enough(cand_lat, cand_lon, placed, km_per_deg_lon, cur_min_sep):
                        placed.append((cand_lat, cand_lon))
                        break
                    if attempts >= 2000:
                        cur_min_sep *= 0.9
                        attempts = 0
            return placed

        red_positions = place_team(red_ids, red_base_lat_deg, red_base_lon_deg, km_per_deg_lon_red, inner_radius_km,
                                   min_sep_km)
        blue_positions = place_team(blue_ids, blue_base_lat_deg, blue_base_lon_deg, km_per_deg_lon_blue,
                                    inner_radius_km, min_sep_km)

        # --- 状态回填 (重点修改：高度 -> 速度) ---

        # 红队初始化
        for idx, sid in enumerate(red_ids):
            sim = self._jsbsims[sid]

            # 1. 随机高度 (5000 - 10000米)
            altitude_m = random.randint(5000, 10000)

            # 2. 随机马赫数 (0.6 - 0.9)
            # 解释：这是亚音速到跨音速的区间，飞机在这个区间升力足够且不易解体
            if altitude_m > 8000:
                # 高空：需要更快一点来保持升力 (0.7 - 0.95)
                target_mach = random.uniform(0.7, 0.95)
            else:
                # 中低空：空气稠密，慢一点没事 (0.6 - 0.9)
                target_mach = random.uniform(0.6, 0.9)

            # 3. 根据高度和马赫数，算出匹配的真速 (m/s)
            speed_mps = get_speed_from_mach(altitude_m, target_mach)

            # 4. 航向
            # heading_noise = random.uniform(-30, 30)
            heading_noise = random.uniform(0, 360)

            heading_deg = (red_ideal_heading + heading_noise) % 360

            lat_deg, lon_deg = red_positions[idx]
            sim.reload({
                "ic_long_gc_deg": lon_deg,
                "ic_lat_geod_deg": lat_deg,
                "ic_h_sl_ft": altitude_m * FT_PER_METER,
                "ic_psi_true_deg": heading_deg,
                "ic_u_fps": speed_mps * FT_PER_METER,  # 注意单位转换
            })

        # 蓝队初始化
        for idx, sid in enumerate(blue_ids):
            sim = self._jsbsims[sid]

            # 1. 随机高度
            altitude_m = random.randint(5000, 10000)

            # 2. 随机马赫数 (保持一致的物理区间)
            target_mach = random.uniform(0.6, 0.9)

            # 3. 算出真速
            speed_mps = get_speed_from_mach(altitude_m, target_mach)

            # 4. 航向
            # heading_noise = random.uniform(-30, 30)
            heading_noise = random.uniform(0, 360)

            heading_deg = (blue_ideal_heading) % 360

            lat_deg, lon_deg = blue_positions[idx]
            sim.reload({
                "ic_long_gc_deg": lon_deg,
                "ic_lat_geod_deg": lat_deg,
                "ic_h_sl_ft": altitude_m * FT_PER_METER,
                "ic_psi_true_deg": heading_deg,
                "ic_u_fps": speed_mps * FT_PER_METER,
            })

        for sid in other_ids:
            raise ValueError(f"Unsupported sim_id prefix for {sid}. Use 'A' or 'B'.")

        self._tempsims.clear()
    def normal_reset_simulators(self):
        # Assign new initial condition here!
        for sim in self._jsbsims.values():
            sim.reload()
        self._tempsims.clear()
        # self.new_random_reset_simulators()
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's observation. Accepts an action and
        returns a tuple (observation, reward_visualize, done, info).

        Args:
            action (dict): the agents' actions, each key corresponds to an agent_id

        Returns:
            (tuple):
                obs: agents' observation of the current environment
                share_obs: agents' share observation of the current environment
                rewards: amount of rewards returned after previous actions
                dones: whether the episode has ended, in which case further step() calls are undefined
                info: auxiliary information
        """
        self.current_step += 1
        info = {"current_step": self.current_step}

        # apply actions
        action = self._unpack(action)
        for agent_id in self.agents.keys():
            a_action = self.task.normalize_action(self, agent_id, action[agent_id])
            self.agents[agent_id].set_property_values(self.task.action_var, a_action)
        # run simulation
        for _ in range(self.agent_interaction_steps):
            for sim in self._jsbsims.values():
                sim.run()
            for sim in self._tempsims.values():
                sim.run()
        self.task.step(self)
        obs = self.get_obs()
        share_obs = self.get_state()

        self._record_step_metrics(info)

        rewards = {}
        for agent_id in self.agents.keys():
            reward, info = self.task.get_reward(self, agent_id, info)
            rewards[agent_id] = [reward]
        ego_reward = np.mean([rewards[ego_id] for ego_id in self.ego_ids])
        enm_reward = np.mean([rewards[enm_id] for enm_id in self.enm_ids])
        for ego_id in self.ego_ids:
            rewards[ego_id] = [ego_reward]
        for enm_id in self.enm_ids:
            rewards[enm_id] = [enm_reward]

        dones = {}
        for agent_id in self.agents.keys():
            done, info = self.task.get_termination(self, agent_id, info)
            dones[agent_id] = [done]

        return self._pack(obs), self._pack(share_obs), self._pack(rewards), self._pack(dones), info

    def _init_episode_metrics(self):
        self.episode_metrics = {
            "step": [],
            "per_agent": {},
            "team": {
                "ego": {
                    "alive_count": [],
                    "mean_speed": [],
                    "mean_specific_energy": [],
                    "mean_geometry_score": [],
                    "mean_favorable_geometry": [],
                    "mean_wez_advantage": [],
                    "mean_wez_dominance": [],
                },
                "enm": {
                    "alive_count": [],
                    "mean_speed": [],
                    "mean_specific_energy": [],
                    "mean_geometry_score": [],
                    "mean_favorable_geometry": [],
                    "mean_wez_advantage": [],
                    "mean_wez_dominance": [],
                }
            }
        }

        for agent_id in self.agents.keys():
            self.episode_metrics["per_agent"][agent_id] = {
                "speed": [],
                "specific_energy": [],
                "geometry_score": [],
                "favorable_geometry": [],
                "wez_advantage": [],
                "wez_dominance": [],
            }

    # def _is_in_wez(self, attacker_id, target_id):
    #     """
    #     TODO: 按你的环境改这里
    #     返回 bool: attacker 是否对 target 形成 WEZ 优势
    #     这里我不给你乱写，因为这部分最依赖你的导弹/雷达/发射逻辑
    #
    #     你可以先用一个临时近似版本：
    #     例如：距离 < 某阈值 且 AO/TA 满足条件
    #     """
    #     raise NotImplementedError(
    #         f"_is_in_wez() 需要按你的 WEZ 判定逻辑适配, attacker={attacker_id}, target={target_id}"
    #     )

    # =========================
    # 3) 通用辅助函数
    # =========================
    def _safe_mean(self, values):
        return float(np.mean(values)) if len(values) > 0 else 0.0

    def _safe_last(self, values):
        return float(values[-1]) if len(values) > 0 else 0.0

    def _select_reference_enemy(self, agent, enemies, mode="nearest"):
        """
        给某个 agent 选一个参考敌机，用来算 AO/TA 几何指标
        默认用最近敌机
        """
        alive_enemies = [enm for enm in enemies if enm.is_alive]
        if len(alive_enemies) == 0:
            return None

        if mode == "nearest":
            my_pos = agent.get_position()
            dists = []
            for enm in alive_enemies:
                enm_pos = enm.get_position()
                d = np.linalg.norm(my_pos - enm_pos)
                dists.append((d, enm))
            dists.sort(key=lambda x: x[0])
            return dists[0][1]

        # 你后面也可以扩展：
        # mode == "assigned_target"
        # mode == "most_threatening"
        return alive_enemies[0]

    def _compute_geometry_against_enemy(self, agent, enm):
        """
        返回:
            geometry_score: float
            favorable_geometry: float(0/1)
            ao: float
            ta: float
            r: float
        依赖你现有的 get_AO_TA_R()
        """
        pos = agent.get_position()
        vel = agent.get_velocity()
        enm_pos = enm.get_position()
        enm_vel = enm.get_velocity()

        ego_feature = np.concatenate([pos, vel])
        enm_feature = np.concatenate([enm_pos, enm_vel])

        # 这里假设你已经有这个函数
        ao, ta, r = get_AO_TA_R(ego_feature, enm_feature)

        # AO 越小越好, TA 越大越好
        s_ao = (1.0 + np.cos(ao)) / 2.0
        s_ta = (1.0 - np.cos(ta)) / 2.0
        geometry_score = 0.5 * s_ao + 0.5 * s_ta

        ao_thr = np.deg2rad(30.0)
        ta_thr = np.deg2rad(120.0)
        favorable_geometry = float((ao < ao_thr) and (ta > ta_thr))

        return float(geometry_score), float(favorable_geometry), float(ao), float(ta), float(r)

    def _compute_agent_wez_metrics(self, agent_id, enemy_ids):
        """
        单机 WEZ 指标:
        - self_in_wez: 我能否打任一敌机
        - opp_in_wez:  任一敌机能否打我
        - wez_advantage: 我能打人且别人不能打我
        - wez_dominance: self_in_wez - opp_in_wez
        """
        alive_enemy_ids = [eid for eid in enemy_ids if self._is_alive(eid)]
        if len(alive_enemy_ids) == 0:
            return 0.0, 0.0, 0.0, 0.0

        self_in_wez = 0.0
        opp_in_wez = 0.0

        for eid in alive_enemy_ids:
            if self._is_in_wez(agent_id, eid):
                self_in_wez = 1.0
            if self._is_in_wez(eid, agent_id):
                opp_in_wez = 1.0

        wez_advantage = float((self_in_wez == 1.0) and (opp_in_wez == 0.0))
        wez_dominance = float(self_in_wez - opp_in_wez)
        return float(self_in_wez), float(opp_in_wez), wez_advantage, wez_dominance

    def _compute_specific_energy(self, agent):
        """
        比能/能量高度:
            h_e = h + V^2 / (2g)

        假设位置是 NED:
            down 为正 -> 高度 h = -down
        """
        pos = agent.get_position()
        vel = agent.get_velocity()

        v = float(np.linalg.norm(vel))
        h = float(pos[2])  # 如果你的 z 不是 down，这里自己改
        he = float(h + v * v / (2.0 * 9.81))
        return v, he

    # =========================
    # 4) 每一步统计：核心函数
    # =========================
    def _record_step_metrics(self, info):
        """
        这个函数放在:
            self.task.step(self)
            obs = self.get_obs()
            share_obs = self.get_state()
        之后
        """

        if not hasattr(self, "episode_metrics") or self.episode_metrics is None:
            self._init_episode_metrics()

        self.episode_metrics["step"].append(int(self.current_step))

        # ---------- 先记录每架飞机 ----------
        for agent_id in self.agents.keys():
            # 死亡 agent 不记当前 step
            agent = self.agents[agent_id]
            if not agent.is_alive:
                continue

            # 速度 / 比能
            speed, he = self._compute_specific_energy(agent)
            self.episode_metrics["per_agent"][agent_id]["speed"].append(speed)
            self.episode_metrics["per_agent"][agent_id]["specific_energy"].append(he)

            # 选择敌方集合
            if agent_id in self.ego_ids:
                enemy_ids = self.enm_ids
            else:
                enemy_ids = self.ego_ids

            # 几何指标：相对于一个参考敌机
            ref_enemy = self._select_reference_enemy(agent, agent.enemies, mode="nearest")
            if ref_enemy is None:
                geometry_score = 0.0
                favorable_geometry = 0.0
                ao = 0.0
                ta = 0.0
                r = 0.0
            else:
                geometry_score, favorable_geometry, ao, ta, r = self._compute_geometry_against_enemy(
                    agent, ref_enemy
                )

            self.episode_metrics["per_agent"][agent_id]["geometry_score"].append(geometry_score)
            self.episode_metrics["per_agent"][agent_id]["favorable_geometry"].append(favorable_geometry)

            # WEZ 指标：对任一敌机
            # _, _, wez_advantage, wez_dominance = self._compute_agent_wez_metrics(agent_id, enemy_ids)
            # self.episode_metrics["per_agent"][agent_id]["wez_advantage"].append(wez_advantage)
            # self.episode_metrics["per_agent"][agent_id]["wez_dominance"].append(wez_dominance)

        # ---------- 再做 team 聚合 ----------
        self._aggregate_team_step_metrics(side="ego")
        self._aggregate_team_step_metrics(side="enm")

        # ---------- 可选：把本 step 的 team 指标塞进 info ----------
        info["step_metrics"] = {
            "ego_mean_speed": self._safe_last(self.episode_metrics["team"]["ego"]["mean_speed"]),
            "ego_mean_specific_energy": self._safe_last(self.episode_metrics["team"]["ego"]["mean_specific_energy"]),
            "ego_mean_geometry_score": self._safe_last(self.episode_metrics["team"]["ego"]["mean_geometry_score"]),
            "ego_mean_wez_advantage": self._safe_last(self.episode_metrics["team"]["ego"]["mean_wez_advantage"]),
            "ego_mean_wez_dominance": self._safe_last(self.episode_metrics["team"]["ego"]["mean_wez_dominance"]),

            "enm_mean_speed": self._safe_last(self.episode_metrics["team"]["enm"]["mean_speed"]),
            "enm_mean_specific_energy": self._safe_last(self.episode_metrics["team"]["enm"]["mean_specific_energy"]),
            "enm_mean_geometry_score": self._safe_last(self.episode_metrics["team"]["enm"]["mean_geometry_score"]),
            "enm_mean_wez_advantage": self._safe_last(self.episode_metrics["team"]["enm"]["mean_wez_advantage"]),
            "enm_mean_wez_dominance": self._safe_last(self.episode_metrics["team"]["enm"]["mean_wez_dominance"]),
        }

    def _aggregate_team_step_metrics(self, side="ego"):
        if side == "ego":
            side_agent_ids = self.ego_ids
        else:
            side_agent_ids = self.enm_ids

        alive_ids = [aid for aid in side_agent_ids if self.agents[aid].is_alive]

        team_buf = self.episode_metrics["team"][side]
        team_buf["alive_count"].append(len(alive_ids))

        if len(alive_ids) == 0:
            team_buf["mean_speed"].append(0.0)
            team_buf["mean_specific_energy"].append(0.0)
            team_buf["mean_geometry_score"].append(0.0)
            team_buf["mean_favorable_geometry"].append(0.0)
            team_buf["mean_wez_advantage"].append(0.0)
            team_buf["mean_wez_dominance"].append(0.0)
            return

        speeds = []
        hes = []
        geos = []
        favs = []
        wez_advs = []
        wez_doms = []

        for aid in alive_ids:
            agent_buf = self.episode_metrics["per_agent"][aid]
            speeds.append(self._safe_last(agent_buf["speed"]))
            hes.append(self._safe_last(agent_buf["specific_energy"]))
            geos.append(self._safe_last(agent_buf["geometry_score"]))
            favs.append(self._safe_last(agent_buf["favorable_geometry"]))
            wez_advs.append(self._safe_last(agent_buf["wez_advantage"]))
            wez_doms.append(self._safe_last(agent_buf["wez_dominance"]))

        team_buf["mean_speed"].append(self._safe_mean(speeds))
        team_buf["mean_specific_energy"].append(self._safe_mean(hes))
        team_buf["mean_geometry_score"].append(self._safe_mean(geos))
        team_buf["mean_favorable_geometry"].append(self._safe_mean(favs))
        team_buf["mean_wez_advantage"].append(self._safe_mean(wez_advs))
        team_buf["mean_wez_dominance"].append(self._safe_mean(wez_doms))

    # =========================
    # 5) episode 结束时汇总
    # =========================
    def _finalize_episode_metrics(self):
        result = {
            "per_agent": {},
            "team": {
                "ego": {},
                "enm": {},
            }
        }

        # --- per-agent 汇总 ---
        for agent_id, buf in self.episode_metrics["per_agent"].items():
            result["per_agent"][agent_id] = {
                "mean_speed": self._safe_mean(buf["speed"]),
                "end_speed": self._safe_last(buf["speed"]),

                "mean_specific_energy": self._safe_mean(buf["specific_energy"]),
                "end_specific_energy": self._safe_last(buf["specific_energy"]),

                "geometry_score_mean": self._safe_mean(buf["geometry_score"]),
                "favorable_geometry_ratio": self._safe_mean(buf["favorable_geometry"]),

                "wez_advantage_ratio": self._safe_mean(buf["wez_advantage"]),
                "wez_dominance_mean": self._safe_mean(buf["wez_dominance"]),
            }

        # --- team 汇总 ---
        for side in ["ego", "enm"]:
            tbuf = self.episode_metrics["team"][side]
            result["team"][side] = {
                "alive_count_mean": self._safe_mean(tbuf["alive_count"]),
                "alive_count_end": self._safe_last(tbuf["alive_count"]),

                "mean_speed_over_time": self._safe_mean(tbuf["mean_speed"]),
                "end_mean_speed": self._safe_last(tbuf["mean_speed"]),

                "mean_specific_energy_over_time": self._safe_mean(tbuf["mean_specific_energy"]),
                "end_mean_specific_energy": self._safe_last(tbuf["mean_specific_energy"]),

                "mean_geometry_score_over_time": self._safe_mean(tbuf["mean_geometry_score"]),
                "mean_favorable_geometry_ratio": self._safe_mean(tbuf["mean_favorable_geometry"]),

                "mean_wez_advantage_ratio": self._safe_mean(tbuf["mean_wez_advantage"]),
                "mean_wez_dominance_over_time": self._safe_mean(tbuf["mean_wez_dominance"]),
            }

        return result