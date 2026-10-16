import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

import argparse
import logging
import wandb
import torch

from my_transformer_components import TransformerLM
from utils import load_memmap_dataset, get_batch, cross_entropy, AdamW, cosine_scheduler, gradient_clipping, save_checkpoint, load_checkpoint

def get_args():
    parser = argparse.ArgumentParser()
    # 模型超参
    parser.add_argument("--d_model", type=int, default=1600)
    parser.add_argument("--num_heads", type=int, default=25)
    parser.add_argument("--d_ff", type=int, default=4288)
    parser.add_argument("--theta", type=int, default=10000)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=48)
    # 训练超参
    # 注意type
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_l2_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max_lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--cosine_iters", type=int, default=10000)
    # 数据集信息
    parser.add_argument("--dataset_path", type=str, default="dataset/train.bin")
    # 注意整数id默认uint16/uint32
    parser.add_argument("--dataset_dtype", type=str, default="uint32")
    # 输出路径信息
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Path to checkpoint to resume from; leave None for fresh start")
    parser.add_argument("--use_wandb", action="store_true")
    return parser.parse_args()

def setup_wandb(args):
    if not args.use_wandb:
        return
    wandb.init(
        project="cs336-hw1",
        name=f"run_{args.out_dir}",
        config=vars(args)
    )

def setup_logger(log_file: str):
    """
    配置logger：同时输出终端 + 输出文件
    :param log_file: 日志文件路径，例如 ./outputs/run.log
    :return: logger对象
    """
    # 1. 获取logger实例，参数为名字
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)   # logger接收的最低等级

    # 避免重复添加handler（多次调用setup会重复打印）
    if logger.handlers:
        return logger

    # 2. 定义日志格式，%(asctime)s指时间，%(name)s指logger的名字，%(levelname)s指level，%(message)s指实际内容
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 3. Handler1：输出到控制台
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)  # 控制台只输出INFO及以上，屏蔽DEBUG
    stream_handler.setFormatter(formatter)

    # 4. Handler2：输出写入文件（文件可以记录DEBUG完整信息）
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)   # 文件保存全部DEBUG以上信息
    file_handler.setFormatter(formatter)

    # 挂载handler
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger

def main():
    # 用户参数
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # 设置随机种子
    seed = 2026
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 路径处理
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.log_path is None:
        log_path = out_dir / "train_log.txt"
    else:
        log_path = Path(args.log_path)
    # 日志初始化
    logger = setup_logger(log_path)
    if args.use_wandb:
        setup_wandb(args)
    # 数据集
    dataset = load_memmap_dataset(args.dataset_path, args.dataset_dtype)
    # 模型，注意设置device
    model = TransformerLM(
        d_model=args.d_model, 
        num_heads=args.num_heads, 
        d_ff=args.d_ff, 
        theta=args.theta, 
        vocab_size=args.vocab_size, 
        context_length=args.context_length,
        num_layers=args.num_layers
    ).to(device)
    # 开启训练模式
    model.train()
    # 优化器
    optim = AdamW(model.parameters(), args.lr, args.weight_decay, (args.beta1, args.beta2), 1e-6)
    # 加载
    start_step = 0
    if args.resume_ckpt is not None:
        resume_ckpt_path = out_dir / args.resume_ckpt
        if not resume_ckpt_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_ckpt_path}")
        start_step = load_checkpoint(resume_ckpt_path, model, optim)
        logger.info(f"Resumed from {resume_ckpt_path}, start_step = {start_step}")
    # 训练循环
    for i in range(start_step, args.max_steps):
        batch_data, labels = get_batch(dataset, args.batch_size, args.context_length, device)
        # 注意先计算lr，再执行训练
        current_lr = cosine_scheduler(i, args.max_lr, args.min_lr, args.warmup_iters, args.cosine_iters)
        optim.set_lr(current_lr)
        optim.zero_grad()
        out = model(batch_data)
        loss = cross_entropy(out, labels)
        loss.backward()
        gradient_clipping(model.parameters(), args.max_l2_norm)
        optim.step()
        # 日志 + 保存
        if i % 20 == 0:
            val_ppl = torch.exp(loss)
            log_data = {
                "train/loss": loss.item(),
                "train/lr": optim.get_lr(),
            }
            if args.use_wandb:
                wandb.log(log_data, step=i)
            logger.info(f"step: {i}, train loss: {loss.item():.4f}, train ppl: {val_ppl.item():.2f}")
        if i % args.save_interval == 0:
            ckpt_path = out_dir / f"ckpt_{i:06d}.pt"
            save_checkpoint(model, optim, i, ckpt_path)
            logger.info(f"Saving checkpoint to {ckpt_path}")
    # 注意关闭wandb
    if args.use_wandb:
        wandb.finish()
    logger.info("Training finished.")

if __name__ == '__main__':
    main()