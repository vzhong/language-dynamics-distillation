# README
This repository contains the source code for the paper [Improving Policy Learning via Language Dynamics Distillation](#).

## Citation

```
@inproceedings{ zhong2022ldd,
  title={ Improving Policy Learning via Language Dynamics Distillation },
  author={ Victor Zhong and Jesse Mu and Luke Zettlemoyer and Edward Grefenstette and Tim Rocktäschel },
  booktitle={ NeurIPS },
  year={ 2022 }
}
```

## Preliminaries
This code base consists of 3 modified original code bases to add LDD (e.g. obtaining demonstrations, dynamics modeling, distillation+initialization).
- `silg` is a modification of [silg](https://github.com/vzhong/silg) to incorporate language descriptions into Touchdown, and also add LDD.
- `messenger` is a modification of [messenger-emma](https://github.com/ahjwang/messenger-emma)
- `nethack` is a modification of [nle-sample-factory-baseline](https://github.com/Miffyli/nle-sample-factory-baseline).

## License

We distribute this work under the MIT license. The dataset we use are publically available and
distributed as a part of the SILG benchmark [Zhong et al., 2021]. There are no personally identifying
information in the assets we use. SILG is distributed under a MIT license. The included environments
are licensed as follows:
1. NetHack: NetHack General Public License
2. Touchdown: Creative Commons Attribution 4.0 International
3. ALFWorld: MIT
4. RTFM: Attribution-NonCommercial 4.0 International
5. Messenger: MIT

## Experiments
For launch commands, you need to specify a `--partition` to run on slurm, or `--local` to run locally.

To train a baseline model:

```
# nethack
cd nethack
pip install -r requirements.txt
python train.py gpus=1

# silg rtfm
cd silg
bash install_envs.sh
pip install -e .
python launch.py --envs rtfm

# silg touchdown
cd silg
bash install_envs.sh
pip install -e .
python launch.py --envs touchdown_lang

# silg alfworld
cd silg
bash install_envs.sh
pip install -e .
python launch.py --envs alfworld

# messenger
cd messenger
pip install -r requirements.txt
pip install -e .
python stage_one_emma.py --output saves/emma_s1 --eval_interval 5000 --max_eps 300000 --max_steps 10
python stage_two_emma.py --output saves/emma_s2 --eval_interval 5000 --max_eps 300000 --max_steps 10 --load_state saves/emma_s1_state.pth
```

To collect demonstrations:

```
# nethack
# this (new) ttyrecs data is being released by a third party and will be open-sourced soon.
# https://openreview.net/pdf?id=zHNNSzo10xN
cd nethack/lm
python dump_data.py

# silg rtfm
cd silg
python run_exp.py --mode rollout --resume <path_to_model.tar>

# silg touchdown
cd silg
python rollout_touchdown_lang.py

# silg alfworld
cd silg
python rollout_alfworld.py

# messenger
cd messenger
unzip pretrained.zip
python rollout.py --output saves/expert_s2 --load_state saves/emma_s2_state.pth
```

To train dynamics model:

```
# for silg rtfm, touchdown, and alfworld
cd silg
python train_dynamics.py drollout=<path_to_demonstrations_dir> pretrain=<path_to_rl_exp_dir>

# for messenger
cd messenger
python train_dynamics.py suffix=stage2 frollout=$PWD/saves/expert_s2_rollouts.pt
```

To fine-tune with dynamics model during RL:
```
# for nethack
cd nethack/lm
python train_lm.py lm.model=char_image_dm lm.process_image=true

# for silg rtfm, touchdown, and alfworld
cd silg
# you need to modify the f=*.ckpt paths to point to checkpoints trained in the previous step
python launch_distill.py --envs rtfm touchdown_lang alfworld

# for messenger
cd messenger
python stage_two_emma.py --output saves/emma_s2_distill --eval_interval 5000 --max_eps 300000 --max_steps 10 --pretrain_checkpoint saves/mymodel-stage2/last.ckpt --pretrain distill
```
