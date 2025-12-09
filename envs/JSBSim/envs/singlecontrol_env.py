import numpy as np

from envs.JSBSim.human_task.HumanFreeFlyTask import HumanFreeFlyTask
from .env_base import BaseEnv
from ..tasks.heading_task import HeadingTask


class SingleControlEnv(BaseEnv):
    """
    SingleControlEnv is an fly-control env for single agent with no enemy fighters.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        assert len(self.agents.keys()) == 1, f"{self.__class__.__name__} only supports 1 aircraft!"
        self.init_states = None
        self.curriculum_level = 0.0  # 0.0: 任务最简单, 1.0: 任务最难

        # 定义最大难度界限 (对应论文里的 Hard)
        self.max_heading_diff = 179.0  # 最大允许转弯 179 度
        self.max_alt_diff = 2000.0  # 最大允许高度差 2000 ft
        self.max_speed_diff_ratio = 0.3  # 最大允许速度差异 30%

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'heading':
            self.task = HeadingTask(self.config)
        elif taskname == "HumanFreeFly":
            self.task = HumanFreeFlyTask(self.config)
        else:
            raise NotImplementedError(f'Unknown taskname: {taskname}')

    def reset(self):
        self.current_step = 0
        self.reset_simulators()
        self.heading_turn_counts = 0
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs)

    def set_curriculum_level(self, level):
        """
        供训练主循环调用，更新课程难度
        """
        self.curriculum_level = level
        # 可选：打印一下确认更新收到
        # print(f"Agent {self.agent_id} difficulty updated to {level:.2f}")

    def reset_simulators(self):
        if self.init_states is None:
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]

        # 1. 随机生成初始状态 (保持你原有的逻辑，这很好，增加了鲁棒性)
        init_heading = self.np_random.uniform(0., 360.)  # 建议改成 0-360 覆盖全向
        init_altitude = self.np_random.uniform(14000., 25000.)  # 留出空间给爬升
        init_velocities_u = self.np_random.uniform(500., 900.)  # 留出空间给加速减速

        for init_state in self.init_states:
            # 2. 根据课程难度 (self.curriculum_level) 生成偏差
            # 难度越低，deviation 越接近 0；难度越高，deviation 波动越大

            # --- 航向偏差 ---
            # 难度0时: 偏差0度; 难度1时: 偏差 -179 ~ +179 度
            heading_deviation = self.np_random.uniform(-1, 1) * self.max_heading_diff * self.curriculum_level
            target_heading = (init_heading + heading_deviation) % 360.0

            # --- 高度偏差 ---
            # 难度0时: 偏差0ft; 难度1时: 偏差 -2000 ~ +2000 ft
            alt_deviation = self.np_random.uniform(-1, 1) * self.max_alt_diff * self.curriculum_level
            target_altitude = init_altitude + alt_deviation
            # 保护逻辑：防止目标高度钻地或太高
            target_altitude = np.clip(target_altitude, 5000., 35000.)

            # --- 速度偏差 ---
            # 难度1时: 速度改变 +/- 30%
            speed_deviation_ratio = self.np_random.uniform(-1, 1) * self.max_speed_diff_ratio * self.curriculum_level
            target_velocities_u = init_velocities_u * (1.0 + speed_deviation_ratio)

            # 更新初始状态
            init_state.update({
                'ic_psi_true_deg': init_heading,
                'ic_h_sl_ft': init_altitude,
                'ic_u_fps': init_velocities_u,

                # 关键：目标状态不再等于初始状态，而是有了受控的偏差
                'target_heading_deg': target_heading,
                'target_altitude_ft': target_altitude,
                'target_velocities_u_mps': target_velocities_u * 0.3048,  # fps 转 mps
            })

        for idx, sim in enumerate(self.agents.values()):
            sim.reload(self.init_states[idx])
        self._tempsims.clear()