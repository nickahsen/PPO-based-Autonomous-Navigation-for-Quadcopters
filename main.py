from pathlib import Path

import time

import yaml

import typer

from typing_extensions import Annotated

import gymnasium

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback

from scripts.airsim_env import TrainConfig, ActionType

try:
    import wandb  # pylint: disable=import-outside-toplevel
    from wandb.integration.sb3 import (  # pylint: disable=import-outside-toplevel
        WandbCallback,
    )
except ImportError:
    wandb, WandbCallback = None, None

app = typer.Typer()

current_file_path = Path(__file__).parent


@app.command()
def evaluate(
    config_file: Annotated[Path, typer.Option()] = Path(
        current_file_path / "scripts/config.yml"
    ),
    sim_ip: Annotated[str, typer.Option()] = "127.0.0.1",
    run_steps: Annotated[int, typer.Option()] = 100,
):
    # Get train environment configs
    with open(config_file, "r", encoding="utf8") as f:
        env_config = yaml.safe_load(f)
        config = TrainConfig(**env_config["TrainEnv"])

    env = make_env(sim_ip, config)

    # Wrap env as VecTransposeImage (Channel last to channel first)
    env = VecTransposeImage(env)

    # Load an existing model
    model = PPO.load(
        env=env,
        path="/home/nick/Dev/AirSim/PPO-based-Autonomous-Navigation-for-Quadcopters/best_model",
    )

    num_success = 0
    num_timeout = 0
    num_collision = 0
    # Run the trained policy
    for _ in range(run_steps):
        obs = env.reset()
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = env.step(action)
            if done[0]:
                if info[0]["is_success"]:
                    num_success += 1
                if info[0]["is_collision"]:
                    num_collision += 1
                if info[0]["timeout"]:
                    num_timeout += 1

    print(f"number of success = {num_success}")
    print(f"number of collision = {num_collision}")
    print(f"number of timeout = {num_timeout}")


def make_env(sim_ip: str, config: TrainConfig):
    return DummyVecEnv(
        [
            lambda: Monitor(
                gymnasium.make(
                    "scripts:airsim-env-v0",
                    ip_address=sim_ip,
                    env_config=config,
                    action_type=ActionType.CONTINUOUS_VELOCITY,
                    sim_dt=1.0,
                )
            )
        ]
    )


@app.command()
def train(
    config_file: Annotated[Path, typer.Option()] = Path(
        current_file_path / "scripts/config.yml"
    ),
    sim_ip: Annotated[str, typer.Option()] = "127.0.0.1",
    train_timesteps: Annotated[int, typer.Option()] = 400000,
    seed: Annotated[int, typer.Option()] = 5,
    use_wandb: Annotated[bool, typer.Option()] = False,
):
    # Get train environment configs
    with open(config_file, "r", encoding="utf8") as f:
        env_config = yaml.safe_load(f)
        config = TrainConfig(**env_config["TrainEnv"])

    env = make_env(sim_ip, config)

    # Initialize PPO
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        seed=seed,
        device="cuda",
        tensorboard_log="./tb_logs/",
    )

    eval_callback = EvalCallback(
        env,
        callback_on_new_best=None,
        n_eval_episodes=20,
        best_model_save_path=".",
        log_path=".",
        eval_freq=5000,
    )
    callbacks = [eval_callback]

    if use_wandb:
        if wandb is None:
            print("W&B is not installed, please install it.")

        wandb.init(project="drone_nav", sync_tensorboard=True)  # type: ignore
        callbacks.append(WandbCallback(verbose=2))

    log_name = "ppo_run_" + str(time.time())

    model.learn(
        total_timesteps=train_timesteps, tb_log_name=log_name, callback=callbacks
    )

    # Save policy weights
    model.save("ppo_navigation_policy")


@app.command()
def test(
    config_file: Annotated[Path, typer.Option()] = Path(
        "/home/nick/Dev/AirSim/PPO-based-Autonomous-Navigation-for-Quadcopters/scripts/config.yml"
    ),
    sim_ip: Annotated[str, typer.Option()] = "127.0.0.1",
):
    # Get train environment configs
    with open(config_file, "r", encoding="utf8") as f:
        env_config = yaml.safe_load(f)
        config = TrainConfig(**env_config["TrainEnv"])

    env = gymnasium.make("scripts:airsim-env-v0", ip_address=sim_ip, env_config=config)

    for _ in range(30):
        env.reset()
        env.step([0, 0, 1])
        time.sleep(5)


if __name__ == "__main__":
    app()
