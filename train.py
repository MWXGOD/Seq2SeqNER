import os, json, time, torch, argparse, swanlab
from tool import *
from tqdm import tqdm
from datetime import datetime
from lightning.pytorch import seed_everything
from lightningmodel import Seq2SeqNERModel
from loguru import logger
from accelerate import Accelerator
from torch.optim import AdamW
from data_module import Seq2SeqNERDataModule
from transformers import get_constant_schedule_with_warmup

# 目前存在的主要训练问题：
# 1) forward 没在 bf16 autocast 里
# 2) 每 step 都 swanlab.log（非常拖）
# 3) 清梯度用 set_to_none=True
# 4) 用 accelerate 自带裁剪（避免不必要同步/不匹配）
# 5) 验证：你到底有没有跑在 bf16？
# 


parser = argparse.ArgumentParser()
parser.add_argument('--args_path', type=str, default='argsfile/aishell_ner_args_4_bart.json')
shell_args = parser.parse_args()
args_dict = read_json_args(shell_args.args_path)
hyperargs = Hyperargs(**args_dict)
seed_everything(hyperargs.seed, workers=True)

time_str = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
swanlab.init(
    project=hyperargs.swanlab_project_name,
    config=hyperargs.__dict__,
    experiment_name=f"MWX-Seq2Seq-Demo-{time_str}"
)

# 数据
data_module = Seq2SeqNERDataModule(**hyperargs.__dict__)
data_module.setup(stage = "fit")
train_dataloader = data_module.train_dataloader()
dev_dataloader = data_module.dev_dataloader()

# 模型
model = Seq2SeqNERModel(**hyperargs.__dict__)

# 优化器
optimizer = AdamW(model.parameters(), lr=hyperargs.learning_rate, weight_decay = hyperargs.weight_decay)

# 学习率调度器
num_warmup_steps = max(1, int(hyperargs.warmup_rate * hyperargs.epochs_num * len(train_dataloader)))
scheduler = get_constant_schedule_with_warmup(optimizer = optimizer, num_warmup_steps = num_warmup_steps)

# 混合精度
accelerator = Accelerator(mixed_precision=hyperargs.mixed_precision)
model, optimizer, train_dataloader, dev_dataloader, scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, dev_dataloader, scheduler
)

max_f1 = 0
for epoch in range(hyperargs.epochs_num):
    train_start = time.time()
    model.train()
    train_bar = tqdm(
        train_dataloader,
        desc=f"Epoch [{epoch+1}/{hyperargs.epochs_num}] Training",
        leave=False
    )
    loss_per_epoch = 0
    for batch in train_bar:
        loss = model.training_step(batch, optimizer, scheduler, accelerator)
        loss_per_epoch += loss.item()
    swanlab.log({"train_loss_per_step": loss_per_epoch/len(train_dataloader)})
    train_end = time.time()
    logger.info(f"训练结束，第{epoch+1}轮总时长：{(train_end-train_start)/60:.2f}分钟")
    

    dev_start = time.time()
    model.eval()
    model.on_validation_epoch_start()
    gen_text_per_epoch = []
    lab_text_per_epoch = []
    with torch.no_grad():
        dev_bar = tqdm(
            dev_dataloader,
            desc=f"Epoch [{epoch+1}/{hyperargs.epochs_num}] Validation",
            leave=False
        )
        for batch in dev_bar:
            gen_text_batch, lab_text_batch = model.validation_step(batch)
            gen_text_per_epoch += gen_text_batch
            lab_text_per_epoch += lab_text_batch
    P, R, F1, P_S, R_S, F1_S = model.on_validation_epoch_end()
    swanlab.log({"F1": F1, "F1_S": F1_S})
    final_f1 = max(F1, F1_S)

    os.makedirs(hyperargs.output_result_path, exist_ok=True)
    with open(f"{hyperargs.output_result_path}/gen_text_batch_{epoch}.json", 'w', encoding='utf-8') as f:
        json.dump({"pred_label": gen_text_per_epoch}, f, indent=4, ensure_ascii=False)
    with open(f"{hyperargs.output_result_path}/lab_text_batch_{epoch}.json", 'w', encoding='utf-8') as f:
        json.dump({"gold_label": lab_text_per_epoch}, f, indent=4, ensure_ascii=False)

    if max_f1<final_f1:
        max_f1 = final_f1
        os.makedirs(hyperargs.output_model_path, exist_ok=True)
        save_path = os.path.join(hyperargs.output_model_path, f"{hyperargs.output_model_path.split('/')[-1]}.bin")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hparams": dict(model.hparams),
            }, 
            save_path
        )
        model.seq2seq.save_pretrained(hyperargs.output_model_path)
        model.tokenizer.save_pretrained(hyperargs.output_model_path)
        os.makedirs(hyperargs.output_result_path, exist_ok=True)
        with open(f"{hyperargs.output_result_path}/best_gen_text_batch.json", 'w', encoding='utf-8') as f:
            json.dump({"pred_label": gen_text_per_epoch}, f, indent=4, ensure_ascii=False)
        with open(f"{hyperargs.output_result_path}/best_lab_text_batch.json", 'w', encoding='utf-8') as f:
            json.dump({"gold_label": lab_text_per_epoch}, f, indent=4, ensure_ascii=False)
        logger.info("模型已保存")
    logger.info(f"评价指标F: {final_f1:.2f}")
    dev_end = time.time()
    logger.info(f"验证结束，第{epoch+1}轮总时长：{(dev_end-dev_start)/60:.2f}分钟")
    


















