"""Rollout from a saved PPO checkpoint — saves MP4 and/or data CSV."""

import os
import json
import argparse
import inspect
import numpy as np
import mujoco
import imageio
import pandas as pd
import jax
jax.config.update("jax_default_matmul_precision", "tensorfloat32")
import jax.numpy as jp
import env as _env  # registers custom_ant on import
from brax import envs
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from config import ENV_NAME, PPOConfig
from env import DEFAULT_XML


def _render_video(trajectory, dt, output_path, width=640, height=480,
                  cam_distance=5.0, show_axis=False, xml_path=None):
    fps = max(1, round(1.0 / float(dt)))
    mj_model = mujoco.MjModel.from_xml_path(xml_path or DEFAULT_XML)
    mj_data = mujoco.MjData(mj_model)
    torso_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = torso_id
    cam.distance = cam_distance
    cam.azimuth = 90
    cam.elevation = -15

    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    frames = []
    for ps in trajectory:
        mj_data.qpos[:] = np.asarray(ps.q)
        mj_data.qvel[:] = np.asarray(ps.qd)
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data, camera=cam)

        if show_axis and renderer.scene.ngeom < renderer.scene.maxgeom:
            g = renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                np.array([1, 0, 0, 1], dtype=np.float32),
            )
            mujoco.mjv_connector(
                g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.02,
                np.array([-200, 0, 0.01]),
                np.array([200, 0, 0.01]),
            )
            renderer.scene.ngeom += 1

        frames.append(renderer.render().copy())
    renderer.close()
    imageio.mimsave(output_path, frames, fps=fps)


def _save_data(pipeline_states, dt, steps, output_path, sys, target_speed=None):
    link_names = list(sys.link_names)
    torso_idx  = link_names.index("torso")

    mj_model = mujoco.MjModel.from_xml_path(DEFAULT_XML)

    def _site_id(name):
        return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name)

    foot_site_ids = {
        "foot_1": _site_id("foot_site_1"),
        "foot_2": _site_id("foot_site_2"),
        "foot_3": _site_id("foot_site_3"),
        "foot_4": _site_id("foot_site_4"),
    }

    q        = np.asarray(pipeline_states.q)           # (steps, nq)
    qd       = np.asarray(pipeline_states.qd)          # (steps, nv)
    pos      = np.asarray(pipeline_states.x.pos)       # (steps, nlinks, 3)
    site_pos = np.asarray(pipeline_states.site_xpos)   # (steps, nsites, 3)

    foot_pos = {
        name: site_pos[:, sid, :]
        for name, sid in foot_site_ids.items()
    }

    t = np.arange(steps) * float(dt)

    data = {
        "t": t,
        # Torso pose (from free joint)
        "torso_x":  q[:, 0], "torso_y":  q[:, 1], "torso_z":  q[:, 2],
        "torso_qw": q[:, 3], "torso_qx": q[:, 4],
        "torso_qy": q[:, 5], "torso_qz": q[:, 6],
        # Joint angles
        "hip_1": q[:, 7],  "ankle_1": q[:, 8],
        "hip_2": q[:, 9],  "ankle_2": q[:, 10],
        "hip_3": q[:, 11], "ankle_3": q[:, 12],
        "hip_4": q[:, 13], "ankle_4": q[:, 14],
        # Torso velocities (from free joint qvel)
        "torso_vel_x": qd[:, 0], "torso_vel_y": qd[:, 1], "torso_vel_z": qd[:, 2],
        "torso_ang_x": qd[:, 3], "torso_ang_y": qd[:, 4], "torso_ang_z": qd[:, 5],
        # Joint velocities
        "hip_1_vel": qd[:, 6],  "ankle_1_vel": qd[:, 7],
        "hip_2_vel": qd[:, 8],  "ankle_2_vel": qd[:, 9],
        "hip_3_vel": qd[:, 10], "ankle_3_vel": qd[:, 11],
        "hip_4_vel": qd[:, 12], "ankle_4_vel": qd[:, 13],
        # Torso world position (from Brax x.pos)
        "torso_body_x": pos[:, torso_idx, 0],
        "torso_body_y": pos[:, torso_idx, 1],
        "torso_body_z": pos[:, torso_idx, 2],
        # Target speed (constant for this rollout)
        "target_speed": np.full(steps, target_speed if target_speed is not None else np.nan),
    }

    contact_threshold = 0.10  # foot sphere radius is 0.08m, center at ~0.08m during stance
    for name, fp in foot_pos.items():
        data[f"{name}_x"] = fp[:, 0]
        data[f"{name}_y"] = fp[:, 1]
        data[f"{name}_z"] = fp[:, 2]
        data[f"{name}_contact"] = (fp[:, 2] < contact_threshold).astype(float)
        data[f"{name}_vx"] = np.gradient(fp[:, 0], float(dt))
        data[f"{name}_vy"] = np.gradient(fp[:, 1], float(dt))
        data[f"{name}_vz"] = np.gradient(fp[:, 2], float(dt))

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}  ({len(df)} rows, {len(df.columns)} columns)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rollout from checkpoint")
    p.add_argument("name",             help="Experiment name — loads from runs/<name>/")
    p.add_argument("--checkpoint",     default=None,
                   help="Override checkpoint path (default: runs/<name>/checkpoints/final)")
    p.add_argument("--deterministic",  action="store_true",
                   help="Use deterministic actions instead of sampling")
    p.add_argument("--seed",           type=int, default=0)
    p.add_argument("--target-speed",   type=float, default=1.5,
                   help="Target forward speed for rollout (m/s, default 1.5)")
    p.add_argument("--no-video",       action="store_false", dest="video",
                   help="Skip MP4 rendering")
    p.add_argument("--video-width",    type=int, default=640)
    p.add_argument("--video-height",   type=int, default=480)
    p.add_argument("--cam-distance",   type=float, default=5.0)
    p.add_argument("--show-axis",      action="store_true",
                   help="Draw a red x-axis line at ground level")
    p.add_argument("--xml",            default=None,
                   help="XML for video rendering (default: morphologies/ant.xml)")
    p.add_argument("--no-save-data",   action="store_false", dest="save_data",
                   help="Skip saving trajectory data to runs/<name>/data.csv")
    # Action noise injects Gaussian noise into policy outputs to simulate real-world variability.
    # This separates feedback vs feedforward R² curves which overlap in deterministic rollouts.
    # To revert: remove --noise from your command, or set --noise 0.0
    p.add_argument("--noise",          type=float, default=0.0,
                   help="Std of Gaussian noise added to actions (default: 0.0 = no noise, try 0.1)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = PPOConfig()

    run_dir   = os.path.abspath(os.path.join(cfg.runs_dir, args.name))
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = args.checkpoint or os.path.join(run_dir, "checkpoints", "final")

    env = envs.get_environment(ENV_NAME)

    print(f"Loading policy from {ckpt_path} ...")
    # load_config crashes when a kernel_init_fn is stored as None in the checkpoint
    # JSON (Brax tries KERNEL_INITIALIZER[None] which is a KeyError). Strip those
    # None entries before loading so Brax falls back to its defaults.
    _cfg_file = os.path.join(ckpt_path, "ppo_network_config.json")
    with open(_cfg_file) as f:
        _cfg_dict = json.load(f)
    _nfk = _cfg_dict.get('network_factory_kwargs', {})
    for _k in list(_nfk.keys()):
        if _k.endswith('_init_fn') and _nfk[_k] is None:
            _nfk.pop(_k)
    with open(_cfg_file, 'w') as f:
        json.dump(_cfg_dict, f)
    config = ppo_checkpoint.load_config(ckpt_path)

    params = ppo_checkpoint.load(ckpt_path)
    normalize = (running_statistics.normalize if config.normalize_observations
                 else lambda x, y: x)
    valid_kwargs = inspect.signature(ppo_networks.make_ppo_networks).parameters
    network_kwargs = {k: v for k, v in config.network_factory_kwargs.items()
                     if k in valid_kwargs}
    ppo_network = ppo_networks.make_ppo_networks(
        env.observation_size,
        config.action_size,
        preprocess_observations_fn=normalize,
        **network_kwargs,
    )
    inference_fn = ppo_networks.make_inference_fn(ppo_network)(
        params, deterministic=args.deterministic
    )
    jit_inference = jax.jit(inference_fn)
    jit_step      = jax.jit(env.step)

    rng   = jax.random.PRNGKey(args.seed)
    state = jax.jit(env.reset)(rng)
    state = state.replace(info={'target_speed': jp.array(args.target_speed)})


    noise_scale = args.noise

    def scan_step(carry, _):
        state, rng = carry
        rng, rng_act, rng_noise = jax.random.split(rng, 3)
        action, _ = jit_inference(state.obs, rng_act)
        if noise_scale > 0.0:
            action = action + jax.random.normal(rng_noise, action.shape) * noise_scale
        next_state = jit_step(state, action)
        return (next_state, rng), state.pipeline_state

    steps = cfg.rollout_steps
    print(f"Running {steps}-step rollout ...")
    (_, _), pipeline_states = jax.lax.scan(
        scan_step, (state, rng), None, length=steps
    )

    if args.save_data:
        _save_data(pipeline_states, env.dt, steps,
                   os.path.join(run_dir, "data.csv"), env.sys, args.target_speed)

    if args.video:
        out = os.path.join(run_dir, "rollout.mp4")
        print(f"Rendering {steps} frames to {out} ...")
        trajectory = [
            jax.tree_util.tree_map(lambda x, i=i: x[i], pipeline_states)
            for i in range(steps)
        ]
        _render_video(trajectory, env.dt, out,
                      width=args.video_width, height=args.video_height,
                      cam_distance=args.cam_distance, show_axis=args.show_axis,
                      xml_path=args.xml)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
