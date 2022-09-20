'''
Script for training models on stage 1, where the agent either starts with or with the goal
and the sole objective is to interact with the correct item.
'''

import copy
import argparse
import tqdm
import random

import torch
from transformers import AutoModel, AutoTokenizer

from messenger.models.emma import EMMA
from messenger.utils.train import PPO, ObservationBuffer
from messenger.models.utils import Encoder
from messenger.envs.stage_two import StageTwo


ENCODER_TYPE = 'bert-base-uncased'


def do_rollout(ppo, env, encoder, args, num_episodes):
    # Observation Buffer
    buffer = ObservationBuffer(device=args.device, buffer_size=args.hist_len)

    episodes = []
    for i in tqdm.trange(num_episodes):
        # run test
        ppo.policy_old.eval()

        episode = dict(traj=[], text=None, actions=[])
        episodes.append(episode)

        state, text = env.reset()
        text = encoder.encode(text)
        buffer.reset(state)

        episode['text'] = text.cpu()

        # Running policy_old:
        for t in range(args.max_steps):
            inp_state = buffer.get_obs()

            with torch.no_grad():
                action = ppo.policy_old.act(inp_state, text, None)
            state, reward, done, _ = env.step(action)
            if t == args.max_steps - 1 and reward != 1:
                reward = -1.0 # failed to complete objective
                done = True

            episode['traj'].append((copy.deepcopy(inp_state.cpu()), copy.deepcopy(state)))
            episode['actions'].append(action)

            if done:
                break
            buffer.update(state)

        ppo.policy_old.train()
    return episodes


def rollout(args):
    model_kwargs = {
        "hist_len": args.hist_len,
        "n_latent_var": args.latent_vars,
        "emb_dim": args.emb_dim,
    }

    ModelCls = EMMA

    ppo = PPO(
        ModelCls = ModelCls,
        model_kwargs = model_kwargs,
        device = args.device,
        lr = args.lr,
        gamma = args.gamma,
        K_epochs = args.k_epochs,
        eps_clip = args.eps_clip,
        load_state = args.load_state,
    )

    # logging variables
    env = StageTwo('train-mc')

    # Text Encoder
    encoder_model = AutoModel.from_pretrained(ENCODER_TYPE)
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_TYPE)
    encoder = Encoder(model=encoder_model, tokenizer=tokenizer, device=args.device, max_length=36)

    torch.save(do_rollout(ppo, env, encoder, args, args.num_episodes), args.output + '_train_rollouts.pt')
    torch.save(do_rollout(ppo, StageTwo('val'), encoder, args, 3000), args.output + '_val_rollouts.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--output", required=True, type=str, help="Local output file name or path.")
    parser.add_argument("--seed", default=None, type=int, help="Set the seed for the model.")
    parser.add_argument("--device", default=0, type=int, help="cuda device ordinal to train on.")

    # Model arguments
    parser.add_argument("--load_state", default=None, help="Path to model state dict.")
    parser.add_argument("--latent_vars", default=128, type=int, help="Latent model dimension.")
    parser.add_argument("--hist_len", default=3, type=int, help="Length of history used by state buffer")
    parser.add_argument("--emb_dim", default=256, type=int, help="embedding size for text")

    # Environment arguments
    parser.add_argument("--max_steps", default=4, type=int, help="Maximum num of steps per episode")
    parser.add_argument("--step_penalty", default=0.0, type=float, help="negative reward for each step")

    # Training arguments
    parser.add_argument("--update_timestep", default=64, type=int, help="Number of steps before model update")
    parser.add_argument("--lr", default=0.00005, type=float, help="learning rate")
    parser.add_argument("--gamma", default=0.8, type=float, help="discount factor")
    parser.add_argument("--k_epochs", default=4, type=int, help="num epochs to update")
    parser.add_argument("--eps_clip", default=0.1, type=float, help="clip param for PPO")
    parser.add_argument("--max_time", default=12, type=float, help="max train time in hrs")
    parser.add_argument("--freeze_attention", action="store_true", help="Do not update attention weights.")

    # Logging arguments
    parser.add_argument('--num_episodes', default=5000, type=int, help='number of episodes to roll out')
    parser.add_argument('--log_notes', type=str, help="notes to append to the logs.")
    parser.add_argument('--check_script', action='store_true', help="log quickly just to see script runs okay.")

    args = parser.parse_args()

    args.device = torch.device(f"cuda:{args.device}")

    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    rollout(args)
