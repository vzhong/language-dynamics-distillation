'''
Script for training models on stage 1, where the agent either starts with or with the goal
and the sole objective is to interact with the correct item.
'''

import argparse
import torch
import time
import pickle
import random
from datetime import datetime

from transformers import AutoModel, AutoTokenizer

from messenger.models.emma import EMMA, EMMAMemory
from messenger.utils.train import PPO, GameStats, ObservationBuffer
from messenger.models.utils import Encoder
from messenger.envs.stage_one import StageOne
from messenger.envs.wrappers import TwoEnvWrapper


ENCODER_TYPE = 'bert-base-uncased'


def train(args):
    model_kwargs = {
        "hist_len": args.hist_len,
        "n_latent_var": args.latent_vars,
        "emb_dim": args.emb_dim,
    }

    ModelCls = EMMA

    if args.load_dynamics_state:
        from model.mymodel import Model as DynamicsModel
        d = DynamicsModel(argparse.Namespace())
        d.load_from_checkpoint(checkpoint_path=args.load_dynamics_state)
        fdynamics = args.output + '_dynamics.pt'
        torch.save(d.emma.state_dict(), fdynamics)
        args.load_state = fdynamics
        print('saving dynamics checkpoint to {}'.format(fdynamics))

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

    if args.freeze_attention: # do not update attention weights
        raise Exception("please dont")
        ppo.policy.freeze_attention()

    # memory stores all the information needed by PPO to
    # to compute losses and make updates
    memory = EMMAMemory()

    # logging variables
    teststats = []
    runstats = []

    env = TwoEnvWrapper(1, 'train-sc', 'train-mc')
    eval_env = StageOne(split='val')

    # gamestat tracker
    gs_eval = GameStats({-1: 'death', 1: "win"})
    gs_train = GameStats({-1: 'death', 1: "win"})

    # Text Encoder
    encoder_model = AutoModel.from_pretrained(ENCODER_TYPE)
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_TYPE)
    encoder = Encoder(model=encoder_model, tokenizer=tokenizer, device=args.device, max_length=36)

    # Observation Buffer
    buffer = ObservationBuffer(device=args.device, buffer_size=args.hist_len)

    # training variables
    i_episode = 0
    timestep = 0
    max_win = -1
    max_train_win = -1
    start_time = time.time()

    while True: # main training loop
        state, text = env.reset()
        text = encoder.encode(text)
        buffer.reset(state)

        # Episode loop
        for t in range(args.max_steps):
            timestep += 1

            # Running policy_old:
            action = ppo.policy_old.act(buffer.get_obs(), text, memory)
            state, reward, done, _ = env.step(action)
            
            # add the step penalty
            reward -= abs(args.step_penalty)

            # add rewards to memory and stats
            if t == args.max_steps - 1 and reward != 1:
                reward = -1.0 # failed to complete objective
                done = True
                
            gs_train.step(reward)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update the model if its time
            if timestep % args.update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0
                
            if done:
                break
                
            buffer.update(state)

        gs_train.end_of_episode()
        i_episode += 1

        # check if max_time has elapsed
        if time.time() - start_time > 60 * 60 * args.max_time:
            break

        # logging
        if i_episode % args.log_interval == 0:
            print("Episode {} \t {}".format(i_episode, gs_train))
            runstats.append(gs_train.compress())
            
            if gs_train.compress()['win'] > max_train_win:
                torch.save(ppo.policy_old.state_dict(), args.output + "_maxtrain.pth")
                max_train_win = gs_train.compress()['win']
                
            gs_train.reset()

        # run test
        if i_episode % args.eval_interval == 0:
            gs_eval.reset()
            ppo.policy_old.eval()

            for _ in range(args.eval_eps):
                state, text = eval_env.reset()
                text = encoder.encode(text)
                buffer.reset(state)

                # Running policy_old:
                for t in range(args.max_steps):
                    with torch.no_grad():
                        action = ppo.policy_old.act(buffer.get_obs(), text, None)
                    state, reward, done, _ = eval_env.step(action)
                    if t == args.max_steps - 1 and reward != 1:
                        reward = -1.0 # failed to complete objective
                        done = True
                        
                    gs_eval.step(reward)
                    if done:
                        break
                    buffer.update(state)
                gs_eval.end_of_episode()

            ppo.policy_old.train()

            print("TEST: \t {}".format(gs_eval))
            teststats.append(gs_eval.compress(append={"step": gs_train.total_steps}))

            if gs_eval.compress()['win'] > max_win:
                torch.save(ppo.policy_old.state_dict(), args.output + "_max.pth")
                max_win = gs_eval.compress()['win']
                

            # Save metrics
            with open(args.output + "_metrics.pkl", "wb") as file:
                pickle.dump({"test": teststats, "run": runstats}, file)

            # Save model states
            torch.save(ppo.policy_old.state_dict(), args.output + "_state.pth")
            
        if i_episode > args.max_eps:
            break
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--output", required=True, type=str, help="Local output file name or path.")
    parser.add_argument("--seed", default=None, type=int, help="Set the seed for the model.")
    parser.add_argument("--device", default=0, type=int, help="cuda device ordinal to train on.")

    # Model arguments
    parser.add_argument("--load_state", default=None, help="Path to model state dict.")
    parser.add_argument("--load_dynamics_state", default=None, help="Path to model state dict.")
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
    parser.add_argument("--max_time", default=48, type=float, help="max train time in hrs")
    parser.add_argument("--max_eps", default=3.1e6, type=float, help="max training episodes")
    parser.add_argument("--freeze_attention", action="store_true", help="Do not update attention weights.")

    # Logging arguments
    parser.add_argument('--log_interval', default=5000, type=int, help='number of episodes between logging')
    parser.add_argument('--eval_interval', default=25000, type=int, help='number of episodes between eval')
    parser.add_argument('--eval_eps', default=500, type=int, help='number of episodes to run eval')
    parser.add_argument('--log_notes', type=str, help="notes to append to the logs.")
    parser.add_argument('--check_script', action='store_true', help="log quickly just to see script runs okay.")

    args = parser.parse_args()
    
    args.device = torch.device(f"cuda:{args.device}")
    
    if args.check_script:
        args.eval_interval = 50
        args.eval_eps = 50
        args.log_interval = 50
        args.max_eps = 100
    
    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    if not args.check_script:
        with open("experiments.log", "a") as log:
            log.write(f"\n\n****** Beginning run: {args.output} ********\n")
            log.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if args.log_notes:
                log.write(f"\nRun notes:\t{args.log_notes}\n")
            log.write(f"Args: \n {args}")
    
    train(args)
