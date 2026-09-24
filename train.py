"""Train a locomotion policy with PPO on a custom MorphologyEnv (MJX backend)."""

import argparse
import csv
import os
import time
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
import jax
from brax import envs
from brax.training.agents.ppo import train as ppo
from config import ENV_NAME, PPOConfig
from rollout import _render_video, _save_data
import warnings
warnings.filterwarnings('ignore')
jax.config.update("jax_default_matmul_precision", "tensorfloat32")

ROLLOUT_NOISE_SCALE = 0.0



def parse_args() -> argparse.Namespace:
    cfg = PPOConfig()
    p = argparse.ArgumentParser(description="PPO training on MorphologyEnv")
    p.add_argument("name",          help="Experiment name — all outputs go to runs/<name>/")
    p.add_argument("--timesteps",   type=int, default=cfg.num_timesteps)
    p.add_argument("--num-envs",    type=int, default=cfg.num_envs)
    p.add_argument("--no-rollout",  action="store_true", help="Skip post-training rollout")
    p.add_argument("--seed",        type=int, default=cfg.seed)
    p.add_argument("--num-evals",   type=int, default=cfg.num_evals)
    return p.parse_args()

def main():
    args = parse_args()
    cfg = PPOConfig()
    env = envs.get_environment(ENV_NAME)

    print(f"Observation size : {env.observation_size}")
    print(f"Action size      : {env.action_size}")
    print(f"Devices          : {jax.devices()}")
    print(f"num_envs         : {args.num_envs:,}")
    print(f"Total timesteps  : {args.timesteps:,}")
    print("-" * 60)

    # What to show during each progress step
    _t0 = None
    _metrics_path = None
    _metrics_writer = None
    _metrics_file = None

    def progress_fn(num_steps, metrics):
        nonlocal _t0, _metrics_path, _metrics_writer, _metrics_file
        if _t0 is None:
            _t0 = time.time()
            _metrics_path = os.path.join(run_dir, "metrics.csv")
            os.makedirs(run_dir, exist_ok=True)
            _metrics_file = open(_metrics_path, "w", newline="")
            _metrics_writer = csv.writer(_metrics_file)
            _metrics_writer.writerow(["num_steps", "reward", "elapsed"])
        reward  = float(metrics.get("eval/episode_reward", 0))
        elapsed = time.time() - _t0
        pct     = num_steps / args.timesteps * 100
        sps     = num_steps / elapsed if elapsed > 0 else 0
        _metrics_writer.writerow([num_steps, reward, elapsed])
        _metrics_file.flush()
        print(
            f"Steps: {num_steps:>12,}  ({pct:5.1f}%)  |  "
            f"Reward: {reward:8.3f}  |  "
            f"Elapsed: {elapsed:6.1f}s  |  "
            f"SPS: {sps:>10,.0f}"
        )

    # Setting  up directories
    run_dir  = os.path.abspath(os.path.join(cfg.runs_dir, args.name))
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Run directory    : {run_dir}")

    
    # Start timer and training
    t0 = time.time()
    make_inference_fn, params, metrics = ppo.train(
        environment=env,
        num_timesteps=args.timesteps,
        num_envs=args.num_envs,
        batch_size=cfg.batch_size,
        num_minibatches=cfg.num_minibatches,
        episode_length=cfg.episode_length,
        unroll_length=cfg.unroll_length,
        num_updates_per_batch=cfg.num_updates_per_batch,
        learning_rate=cfg.learning_rate,
        discounting=cfg.discounting,
        entropy_cost=cfg.entropy_cost,
        num_evals=args.num_evals,
        normalize_observations=cfg.normalize_observations,
        seed=args.seed,
        progress_fn=progress_fn,
        save_checkpoint_path=ckpt_dir,
    )

    total = time.time() - t0
    if _metrics_file:
        _metrics_file.close()

    # Symlink "final" to the last checkpoint
    ckpt_steps = sorted(
        d for d in os.listdir(ckpt_dir)
        if os.path.isdir(os.path.join(ckpt_dir, d)) and d[0].isdigit()
    )
    if ckpt_steps:
        final_link = os.path.join(ckpt_dir, "final")
        if os.path.islink(final_link):
            os.remove(final_link)
        os.symlink(ckpt_steps[-1], final_link)
        print(f"Final checkpoint : {final_link} -> {ckpt_steps[-1]}")

    print("-" * 60)
    print(f"Done! Total time : {total:.1f}s  ({total/60:.1f} min)")
    print(f"Final reward     : {metrics.get('eval/episode_reward', 0):.3f}")

    if args.no_rollout:
        return

    # Perform rollout and make video
    steps = cfg.rollout_steps
    print(f"\nRunning {steps}-step rollout ...")

    inference_fn  = make_inference_fn(params)
    jit_inference = jax.jit(inference_fn)
    jit_step      = jax.jit(env.step)
    jit_reset     = jax.jit(env.reset)

    def scan_step(carry, _):
        state, rng = carry
        rng, rng_act, rng_noise = jax.random.split(rng, 3)
        action, _ = jit_inference(state.obs, rng_act)
        if ROLLOUT_NOISE_SCALE > 0.0:
            action = action + jax.random.normal(rng_noise, action.shape) * ROLLOUT_NOISE_SCALE
        next_state = jit_step(state, action)
        return (next_state, rng), state.pipeline_state

    master_rng = jax.random.PNRGKey(args.seed)
    for ii in range(5):
        master_rng, reset_rng, scan_rng = jax.random.split(master_rng,3)
        state        = jit_reset(reset_rng)
        target_speed = float(state.info['target_speed'])

        # Extract data from rollout and get states
        (_, _), pipeline_states = jax.lax.scan(
            scan_step, (state, rng), None, length=steps
        )

        # Save the data into csv for processing and using for analysis and visuals
        _save_data(pipeline_states, env.dt, steps,
                os.path.join(run_dir, f"data_run{ii}.csv"), env.sys, target_speed)

    trajectory = [
        jax.tree_util.tree_map(lambda x, i=i: x[i], pipeline_states)
        for i in range(steps)
    ]

    video_path = os.path.join(run_dir, "rollout.mp4")
    print(f"Rendering to {video_path} ...")
    _render_video(trajectory, env.dt, video_path)
    print(f"Saved: {video_path}")


if __name__ == "__main__":
    main()
