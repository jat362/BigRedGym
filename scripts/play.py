"""Play back a policy trained with a supported Q2 physics backend.

Usage:
    uv run scripts/play.py --task pendulum
    uv run scripts/play.py --task mini_cheetah --num_envs 8
    uv run scripts/play.py --task mini_cheetah --load_run May08_12-34-56_ \
                           --checkpoint 1500

Supports MuJoCo CPU/Warp and optional VSim checkpoints.
"""

import argparse
import random

import torch

from gym.utils.helpers import set_seed
from gym.utils.task_registry import task_registry


def get_play_args(argv=None):
    parser = argparse.ArgumentParser(description="Play a policy trained with Q2")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Checkpoint root under logs/ (default: task config value)",
    )
    parser.add_argument(
        "--load_run",
        type=str,
        default=None,
        help="Run directory name under logs/<experiment_name>/ (default: latest)",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=-1,
        help="Model iteration index (default: latest)",
    )
    parser.add_argument(
        "--original_cfg",
        action="store_true",
        help="Load environment and runner configs saved with the selected run.",
    )
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument(
        "--backend",
        type=str,
        default="mujoco",
        choices=["mujoco", "vsim"],
        help="Physics backend (vsim is CUDA-only; start via "
        "`uv run --env-file .env.vsim ...`)",
    )
    parser.add_argument(
        "--viewer_ui",
        action="store_true",
        default=False,
        help="show the MuJoCo viewer's side panels (hidden by default: their "
        "shortcuts bind most letters to visualisation toggles, which fire "
        "alongside keyboard teleop)",
    )
    parser.add_argument(
        "--keyboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keyboard teleop in the MuJoCo viewer (see "
        "gym/utils/interfaces/MujocoKeyboardInterface.py for key bindings). "
        "Use --no-keyboard to disable.",
    )
    parser.add_argument(
        "--stand_test",
        action="store_true",
        default=False,
        help="Debug: fixed standing commands, zero residual actions, "
        "no pushes, gentle zero-velocity spawn.",
    )
    return parser.parse_args(argv)


def setup(args):
    # Register tasks (imports the task classes).
    import gym.envs  # noqa: F401

    env_cfg, train_cfg = task_registry.get_cfgs(
        args.task,
        original_cfg=args.original_cfg,
        experiment_name=args.experiment_name,
        load_run=args.load_run,
    )

    # Play-time overrides — small batch, long episodes, no pushes.
    env_cfg.env.num_envs = args.num_envs
    env_cfg.env.episode_length_s = 50
    if hasattr(env_cfg, "commands"):
        env_cfg.commands.resampling_time = 9999
    if hasattr(env_cfg, "push_robots"):
        env_cfg.push_robots.toggle = not args.stand_test
    env_cfg.init_state.reset_mode = "reset_to_basic"

    if args.stand_test:
        env_cfg.init_state.root_pos_range[2] = [0.34, 0.34]
        env_cfg.init_state.root_vel_range = [[0.0, 0.0]] * 6


    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.show_ui = args.viewer_ui

    if args.seed is not None:
        env_cfg.seed = args.seed
        train_cfg.seed = args.seed
    else:
        env_cfg.seed = random.randint(0, 10000)
        train_cfg.seed = env_cfg.seed

    train_cfg.runner.device = args.device
    train_cfg.runner.resume = True
    if args.experiment_name is not None:
        train_cfg.runner.experiment_name = args.experiment_name
    if args.load_run is not None:
        train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    train_cfg.logging.enable_local_saving = False

    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    task_registry.set_log_dir_name(train_cfg)
    set_seed(env_cfg.seed)

    env = task_registry.make_env(
        args.task,
        env_cfg,
        device=args.device,
        headless=args.headless,
        backend=args.backend,
    )

    runner = task_registry.make_alg_runner(env, train_cfg)
    runner.switch_to_eval()
    return env, runner




def play(env, runner, stand_test=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def set_standing_commands():
        env.commands[:, :3] = 0.0
        env.commands[:, 3] = 0.30

    if stand_test:
        set_standing_commands()
        print("default_dof_pos:", env.default_dof_pos[0].cpu().numpy().round(3))

    t_log, target_log, actual_log = [], [], []
    n_steps = 10 * int(env.max_episode_length)

    try:
        for i in range(n_steps):
            if stand_test:
                set_standing_commands()

            runner.set_actions(
                runner.actor_cfg["actions"],
                runner.get_inference_actions(),
                runner.actor_cfg["disable_actions"],
            )
            if stand_test:
                env.dof_pos_target.zero_()  # policy off, reference pose only

            env.step()
            env.check_exit()

            t_log.append(i * env.dt)
            target_log.append(env.commands[0, 3].item())
            actual_log.append(env.base_height[0, 0].item())

            if stand_test and i % 100 == 0:
                print(f"\nStep {i}")
                print(f"Target height: {env.commands[0, 3].item():.3f} m")
                print(f"Actual height: {env.base_height[0, 0].item():.3f} m")
                print("Actual joints:", env.dof_pos[0].detach().cpu().numpy().round(3))
                print("Applied torques:", env.torques[0].detach().cpu().numpy().round(2))
                print(
                    "Foot contact forces:",
                    env.contact_forces[0, env.feet_indices, :]
                    .detach().cpu().numpy().round(2),
                )
                print(
                    "Projected gravity:",
                    env.projected_gravity[0].detach().cpu().numpy().round(3),
                )
    finally:
        # Runs even if check_exit() ends the loop when you close the window.
        if t_log:
            plt.figure(figsize=(8, 4))
            plt.plot(t_log, target_log, label="commanded height")
            plt.plot(t_log, actual_log, label="actual height")
            plt.xlabel("time (s)")
            plt.ylabel("base height (m)")
            plt.title("Time-Series Plot")
            plt.legend()
            plt.tight_layout()
            plt.savefig("height_tracking.pdf")
            print("saved height_tracking.pdf")
                        
            plt.savefig("height_tracking.png", dpi=200)
            import csv
            with open("height_tracking.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t_s", "target_m", "actual_m"])
                w.writerows(zip(t_log, target_log, actual_log))
            print("saved height_tracking.png and height_tracking.csv")


if __name__ == "__main__":
    args = get_play_args()
    with torch.no_grad():
        env, runner = setup(args)
        # The MuJoCo warp (GPU) backend is headless-only: render() is a no-op and
        # it has no passive viewer, so nothing draws.  Point the user at cpu.
        if (
            args.backend == "mujoco"
            and not args.headless
            and not hasattr(env._backend, "_viewer_overlay_fn")
        ):
            print(
                "WARNING: the MuJoCo warp (GPU) backend has no interactive viewer "
                "— nothing will render. Use --device cpu for playback."
            )
        if args.keyboard and hasattr(env, "commands") and not args.headless:
            # Each viewer has its own input model: MuJoCo dispatches key
            # events via a callback, vlearn polls key state per frame.
            if args.backend == "vsim":
                from gym.utils.interfaces.VsimKeyboardInterface import (
                    VsimKeyboardInterface,
                )

                VsimKeyboardInterface(env)
            else:
                from gym.utils.interfaces.MujocoKeyboardInterface import (
                    MujocoKeyboardInterface,
                )

                MujocoKeyboardInterface(env)
        if hasattr(env, "commands") and not args.headless:
            # Same indicators, different drawing primitives per viewer.
            if args.backend == "vsim":
                from gym.utils.interfaces.VsimCommandVisualizer import (
                    VsimCommandVisualizer,
                )

                VsimCommandVisualizer(env)
            else:
                from gym.utils.interfaces.CommandVisualizer import CommandVisualizer

                CommandVisualizer(env)
        play(env, runner, args.stand_test)
