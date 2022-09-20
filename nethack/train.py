import os
import gym
import hydra
import pprint
from hydra.utils import get_original_cwd
from wrangl.learn.rl import MoolibVtrace


def create_env(flags):
    import nle
    import wrappers
    kwargs = dict(
        savedir=None,
        character=flags.env.character,
        max_episode_steps=flags.env.max_episode_steps,
        observation_keys=['tty_chars', 'tty_colors', 'tty_cursor'],
        penalty_step=flags.env.penalty_step,
        penalty_time=flags.env.penalty_time,
        penalty_mode=flags.env.fn_penalty_step,
        no_progress_timeout=150,
    )
    env = gym.make(flags.env.name, **kwargs)
    env = wrappers.RenderCharImagesWithNumpyWrapper(
        env, font_path=os.path.join(get_original_cwd(), 'Hack-Regular.ttf'), blstats_cursor=False,
    )
    return env


@hydra.main(config_path="conf", config_name="default")
def main(cfg):
    Model = MoolibVtrace.load_model_class(cfg.model, model_dir='model')
    if not cfg.test_only:
        Model.run_train_test(cfg, create_train_env=create_env, create_eval_env=create_env)
    else:
        checkpoint_path = 'last.ckpt'
        eval_envs = Model.create_env_pool(cfg, create_env, override_actor_batches=1)
        test_results = Model.run_test(cfg, eval_envs, checkpoint_path, eval_steps=cfg.eval_steps)
        pprint.pprint(test_results)


if __name__ == "__main__":
    main()
