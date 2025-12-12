import os, json, time, torch, argparse, swanlab
from tool import *
from tqdm import tqdm
from datetime import datetime
from lightning.pytorch import seed_everything
from lightningmodel import Seq2SeqNERModel
from transformers import AutoModelForSeq2SeqLM
from loguru import logger
from accelerate import Accelerator
from torch.optim import AdamW
from data_module import Seq2SeqNERDataModule
from transformers import get_constant_schedule_with_warmup


parser = argparse.ArgumentParser()
parser.add_argument('--args_path', type=str, default='argsfile/aishell_ner_args_4_bart.json')
shell_args = parser.parse_args()
args_dict = read_json_args(shell_args.args_path)
hyperargs = Hyperargs(**args_dict)
seed_everything(hyperargs.seed, workers=True)

parser = argparse.ArgumentParser()
parser.add_argument('--args_path', type=str, default='argsfile/aishell_ner_args_4_bart.json')
shell_args = parser.parse_args()
args_dict = read_json_args(shell_args.args_path)
hyperargs = Hyperargs(**args_dict)
seed_everything(hyperargs.seed, workers=True)

# 读取数据
data_module = Seq2SeqNERDataModule(**hyperargs.__dict__)
data_module.setup(stage = "test")
test_dataloader = data_module.test_dataloader()

# 读取模型
ckpt = torch.load("ner_full.pt", map_location="cpu")
model = Seq2SeqNERModel(**ckpt["hparams"])
model.load_state_dict(ckpt["state_dict"], strict=True)
model.eval()
model.to(hyperargs.gpu_id)

# 推理
with torch.no_grad():
    test_start = time.time()
    test_bar = tqdm(
        test_dataloader,
        desc=f"Testing",
        leave=False
    )
    gen_text_per_epoch = []
    lab_text_per_epoch = []
    for batch in test_bar:
        gen_text_batch, lab_text_batch = model.validation_step(batch)
        gen_text_per_epoch += gen_text_batch
        lab_text_per_epoch += lab_text_batch
    P, R, F1, P_S, R_S, F1_S = model.on_validation_epoch_end()
    test_end = time.time()
    logger.info(f"测试结束，总时长：{(test_end-test_start)/60:.2f}分钟")
    logger.info(f"测试的实体级别F1：{F1}, Span级别的F1{F1_S}")








