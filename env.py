"""Custom MorphologyEnv — drop-in replacement for Brax's Ant.

To swap morphologies, pass a different xml_file or xml_string to the constructor.
"""

import os
import numpy as np
from brax import base
from brax import envs
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jp
import mujoco

from config import ENV_NAME, PPOConfig

_cfg = PPOConfig()

# ── Default XML file path ─────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XML = os.path.join(_DIR, "morphologies/baseline.xml")  #change here when doing new training, or rollout!


class MorphologyEnv(PipelineEnv):
    """Ant-like locomotion env loaded from an XML string.

    Functionally identical to brax.envs.ant.Ant with backend='mjx'.
    To swap morphologies, pass a different xml_string to the constructor.
    """

    def __init__(
        self,
        xml_file: str = DEFAULT_XML,
        xml_string: str | None = None,
        ctrl_cost_weight: float = 0.2,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: tuple[float, float] = (0.2, 1.0),
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        **kwargs,
    ):
        if xml_string is not None:
            sys = mjcf.loads(xml_string)
        else:
            sys = mjcf.load(xml_file)

        # MJX solver config (matches Brax's Ant mjx branch)
        sys = sys.tree_replace({
            'opt.solver': mujoco.mjtSolver.mjSOL_NEWTON,
            'opt.disableflags': mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
            'opt.iterations': 2,
            'opt.ls_iterations': 4,
        })

        kwargs['n_frames'] = kwargs.get('n_frames', 5)
        super().__init__(sys=sys, backend='mjx', **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._speed_min = _cfg.speed_min
        self._speed_max = _cfg.speed_max

        # Foot clearance reward setup — uncomment to enable
        # import numpy as np
        # mj_model = mujoco.MjModel.from_xml_path(xml_file)
        # self._foot_site_ids = np.array([
        #     mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, f"foot_site_{i}") for i in range(1, 5)
        # ])

    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2, rng_speed = jax.random.split(rng, 4)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        target_speed = jax.random.uniform(
            rng_speed, minval=self._speed_min, maxval=self._speed_max
        )

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state, target_speed)

        reward, done, zero = jp.zeros(3)
        metrics = {
            'reward_speed': zero,
            'reward_survive': zero,
            'reward_ctrl': zero,
            'x_velocity': zero,
            'target_speed': zero,
            'speed_error': zero,
            'reward_straight': zero,
            # 'reward_foot_clearance': zero,  # uncomment with foot clearance reward
        }
        return State(pipeline_state, obs, reward, done, metrics,
                     {'target_speed': target_speed})

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state0 = state.pipeline_state
        assert pipeline_state0 is not None
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        target_speed = state.info['target_speed']

        velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
        speed_error = velocity[0] - target_speed
        speed_reward = -jp.square(speed_error) + 0.3 * velocity[0]

        # penalize y drift
        straight_reward = -jp.square(velocity[1]) # not added in final reward, but useful for experimenting

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))


        obs = self._get_obs(pipeline_state, target_speed)
        reward = speed_reward + healthy_reward - ctrl_cost  # + straight_reward 
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        state.metrics.update(
            reward_speed=speed_reward,
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            x_velocity=velocity[0],
            target_speed=target_speed,
            speed_error=speed_error,
            reward_straight=straight_reward,
            # reward_foot_clearance=foot_clearance_reward,  # uncomment with foot clearance reward
        )
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: base.State, target_speed: jax.Array) -> jax.Array:
        qpos = pipeline_state.q
        qvel = pipeline_state.qd

        if self._exclude_current_positions_from_observation:
            qpos = pipeline_state.q[2:]

        return jp.concatenate([qpos, qvel, jp.array([target_speed])])


# Register on import so `envs.get_environment(ENV_NAME)` works.
envs.register_environment(ENV_NAME, MorphologyEnv)
