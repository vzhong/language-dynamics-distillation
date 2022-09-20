import pytorch_lightning as pl
import wandb
import random
import torch
from torch.optim import Adam
from argparse import Namespace
from transformers.modeling_outputs import BaseModelOutput
from transformers import T5Tokenizer, T5ForConditionalGeneration
from .. import evaluation


class WandbTextCallback(pl.Callback):

    def on_validation_end(self, trainer, pl_module):
        table = wandb.Table(columns=['context', 'pred', 'gold'], data=pl_module.val_cache)

        # delete old files
        unwrapped_run = trainer.logger.experiment
        run = wandb.Api().run(unwrapped_run.path)
        for a in run.logged_artifacts():
            a.delete(delete_aliases=True)

        trainer.logger.experiment.log({
          "examples": table,
          "global_step": trainer.global_step
        })


class Model(pl.LightningModule):

    def __init__(self, hackrl_model, name='google/t5-v1_1-small'):
        super().__init__()
        self.save_hyperparameters(hackrl_model.flags)
        self.tokenizer = T5Tokenizer.from_pretrained(name)
        self.topline_encoder = hackrl_model.topline_encoder
        self.bottomline_encoder = hackrl_model.bottomline_encoder
        # self.screen_encoder = hackrl_model.screen_encoder
        self.core = hackrl_model.core
        self.policy = hackrl_model.policy

        self.val_cache = []
        self.max_val_cache = 1000
        self.val_cache_p_record = 0.1

        self.lm = T5ForConditionalGeneration.from_pretrained(name)

    def get_early_stop_metric(self):
        return 'val_rouge', 'max'

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=2e-5)
        return optimizer

    def get_callbacks(self):
        if self.hparams.wandb.enable:
            return [WandbTextCallback()]
        else:
            return []

    @classmethod
    def extract_message(cls, tty_chars):
        # assume first dim is rows
        return tty_chars[0].tobytes().decode(errors='ignore').strip()

    def tokenize_message(self, m):
        return self.tokenizer(m, max_length=self.hparams.lm.max_message_len, padding='max_length', truncation=True, return_tensors='pt')

    def batch_extract_message(self, tty_chars):
        # assume first dim is batch
        return [self.extract_message(tty) for tty in tty_chars]

    def get_inputs(self, inputs, ns=None, autodevice=True):
        if inputs['tty_chars'].ndim == 5:
            # Remove false iterator batch
            for k, v in inputs.items():
                inputs[k] = v.squeeze(0)
        ns = ns or Namespace()
        next_msgs = self.batch_extract_message(inputs['next_tty_chars'].cpu().numpy())
        # next_msgs = ['<extra_id_0> {} <extra_id_1>'.format(m) for m in next_msgs]
        next_ids = self.tokenize_message(next_msgs)

        context = []
        context_strs = []
        for example_chars, example_next in zip(inputs['tty_chars'].cpu().transpose(0, 1).numpy(), next_msgs):
            example_msgs = self.batch_extract_message(example_chars)
            if self.hparams.lm.ignore_empty_context:
                example_msgs = [m for m in example_msgs if m]
            example_context = ' . '.join(example_msgs)
            context.append(example_context)
            context_strs.append('\n'.join(example_msgs))

        ns.tty_chars = inputs['tty_chars']
        ns.next_ids = next_ids.to(self.device)
        ns.context_strs = context_strs

        if autodevice:
            ns.next_ids = ns.next_ids.to(self.device)
        return ns

    def get_forward_args(self, ns):
        T, B, H, W = ns.tty_chars.shape

        topline = ns.tty_chars[..., 0, :]
        top = self.topline_encoder(
            topline.float(memory_format=torch.contiguous_format).view(T * B, -1)
        )

        bottom_line = ns.tty_chars[..., -2:, :]
        bottom = self.bottomline_encoder(
            bottom_line.float(memory_format=torch.contiguous_format).view(T * B, -1)
        )

        screen = torch.zeros(T * B, 512).to(self.device)

        st = torch.cat([bottom, top, screen], dim=1).view(T, B, -1)
        core, _ = self.core(st)
        return dict(encoder_outputs=BaseModelOutput(last_hidden_state=core.transpose(0, 1)))

    def training_step(self, batch, batch_idx, compute_loss=True):
        ns = self.get_inputs(batch)
        tgt_ids = ns.next_ids['input_ids']
        forward_args = self.get_forward_args(ns)
        decoder_input_ids = self.lm._shift_right(tgt_ids)
        loss = self.lm.forward(decoder_input_ids=decoder_input_ids, use_cache=True, labels=tgt_ids, **forward_args).loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, split_name='val'):
        ns = self.get_inputs(batch)
        tgt_ids = ns.next_ids['input_ids']
        forward_args = self.get_forward_args(ns)
        gen_ids = self.lm.generate(
            num_beams=self.hparams.lm.num_beams,
            use_cache=True,
            early_stopping=False,
            max_length=self.hparams.lm.max_message_len,
            **forward_args,
        )

        gen = [self.tokenizer.decode(w, skip_special_tokens=True) for w in gen_ids]
        gold = [self.tokenizer.decode(t, skip_special_tokens=True) for t in tgt_ids]

        rouge = []
        for c, p, g in zip(ns.context_strs, gen, gold):
            # print('EX')
            # print(c)
            # print(p)
            # print(g)
            # print()
            # import pdb; pdb.set_trace()
            rouge.append(evaluation.metric_max_over_ground_truths(evaluation.rouge, p, [g]))
            if random.random() < self.val_cache_p_record:
                self.val_cache.insert(0, (c, p, g))
        self.val_cache = self.val_cache[:self.max_val_cache]
        self.log('{}_rouge'.format(split_name), sum(rouge) / len(rouge))

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, split_name='test')
