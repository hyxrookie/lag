import numpy as np
from .env_base import BaseEnv
from ..tasks import SingleCombatTask, SingleCombatDodgeMissileTask, HierarchicalSingleCombatDodgeMissileTask, \
    HierarchicalSingleCombatShootTask, SingleCombatShootMissileTask, HierarchicalSingleCombatTask
from ..human_task.HumanSingleCombatTask import  HumanSingleCombatTask


class SingleCombatEnv(BaseEnv):
    """
    SingleCombatEnv is an one-to-one competitive environment.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        assert len(self.agents.keys()) == 2, f"{self.__class__.__name__} only supports 1v1 scenarios!"
        self.init_states = None

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'singlecombat':
            self.task = SingleCombatTask(self.config)
        elif taskname == 'hierarchical_singlecombat':
            self.task = HierarchicalSingleCombatTask(self.config)
        elif taskname == 'singlecombat_dodge_missile':
            self.task = SingleCombatDodgeMissileTask(self.config)
        elif taskname == 'singlecombat_shoot':
            self.task = SingleCombatShootMissileTask(self.config)
        elif taskname == 'hierarchical_singlecombat_dodge_missile':
            self.task = HierarchicalSingleCombatDodgeMissileTask(self.config)
        elif taskname == 'hierarchical_singlecombat_shoot':
            self.task = HierarchicalSingleCombatShootTask(self.config)
        elif taskname == 'HumanSingleCombat':
            self.task = HumanSingleCombatTask(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.reset_simulators()
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs)

    def reset_simulators(self):
        # switch side
        # if self.init_states is None:
        #     self.init_states = [sim.init_state.copy() for sim in self.agents.values()]
        # # self.init_states[0].update({
        # #     'ic_psi_true_deg': (self.np_random.uniform(270, 540))%360,
        # #     'ic_h_sl_ft': self.np_random.uniform(17000, 23000),
        # # })
        # init_states = self.init_states.copy()
        # self.np_random.shuffle(init_states)
        # for idx, sim in enumerate(self.agents.values()):
        #     sim.reload(init_states[idx])
        # self._tempsims.clear()
        self.new_random_reset_simulators()

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
        min_base_separation_km = 35.0  # 30km
        max_base_separation_km = 45.0  # 50km

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
            # heading_deg = (red_ideal_heading + heading_noise) % 360
            # 随机航向
            heading_deg = random.uniform(0, 360)
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
            altitude_m = random.randint(6000, 6000)

            # 2. 随机马赫数 (保持一致的物理区间)
            target_mach = random.uniform(0.6, 0.9)

            # 3. 算出真速
            speed_mps = get_speed_from_mach(altitude_m, target_mach)

            # 4. 航向
            heading_noise = random.uniform(-10, 10)
            heading_deg = (blue_ideal_heading + heading_noise) % 360

            # 随机航向
            # heading_deg = random.uniform(0, 360)
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
