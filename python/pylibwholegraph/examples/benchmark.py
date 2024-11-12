import torch.nn as nn
from typing import Tuple
import datetime
import os
import time
import argparse

import apex
import torch
from apex.parallel import DistributedDataParallel as DDP
import pylibwholegraph.torch as wgth
from ..pylibwholegraph.torch.embedding import WholeMemoryEmbedding, WholeMemoryEmbeddingModule

argparser = argparse.ArgumentParser()
wgth.add_distributed_launch_options(argparser)
wgth.add_training_options(argparser)
wgth.add_common_graph_options(argparser)
wgth.add_common_model_options(argparser)
wgth.add_common_sampler_options(argparser)
wgth.add_node_classfication_options(argparser)
wgth.add_dataloader_options(argparser)
argparser.add_argument(
    "--fp16_embedding", action="store_true", dest="fp16_mbedding", default=False, help="Whether to use fp16 embedding"
)
args = argparser.parse_args()

class GatherFn(nn.Module):
    def __init__(
            self, 
            node_embedding: WholeMemoryEmbedding
    ):
        super(GatherFn, self).__init__()
        self.node_embedding = node_embedding
        self.gather_fn = WholeMemoryEmbeddingModule(self.node_embedding)

    def forward(self, ids):
        x_feat = self.gather_fn(ids, force_dtype=torch.float32)
        return x_feat

def train(train_data, model, optimizer, wm_optimizer, global_comm):
    if wgth.get_rank() == 0:
        print("start training...")
    train_dataloader = wgth.get_train_dataloader(
        train_data,
        args.batchsize,
        replica_id=wgth.get_rank(),
        num_replicas=wgth.get_world_size(),
        num_workers=args.dataloaderworkers,
    )


    train_step = 0
    epoch = 0
    loss_fcn = torch.nn.CrossEntropyLoss()
    train_start_time = time.time()
    while epoch < args.epochs:
        for i, (idx, label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            model.train()
            logits = model(idx)
            loss = torch.cat(logits, dim=1).sum()
            loss.backward()
            optimizer.step()
            if wm_optimizer is not None:
                wm_optimizer.step(args.lr * 0.1)
            if wgth.get_rank() == 0 and train_step % 100 == 0:
                print(
                    "[%s] [LOSS] step=%d, loss=%f"
                    % (
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        train_step,
                        loss.cpu().item(),
                    )
                )
            train_step = train_step + 1
        epoch = epoch + 1
    global_comm.barrier()
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    if wgth.get_rank() == 0:
        print(
            "[%s] [TRAIN_TIME] train time is %.2f seconds"
            % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), train_time)
        )
        print(
            "[EPOCH_TIME] %.2f seconds."
            % ((train_end_time - train_start_time) / args.epochs,)
        )

def measure_time(module: nn.Module, input: torch.Tensor, epochs: int = 100) -> Tuple[float, float]:
    # Measure forward time
    start_time = time.time()
    for _ in range(epochs):
        output = module(input)
    forward_time = (time.time() - start_time) / epochs

    # Measure backward time
    start_time = time.time()
    for _ in range(epochs):
        output = module(input)
        output.sum().backward()
    backward_time = (time.time() - start_time) / epochs

    return forward_time, backward_time

def main(argv):
    print(f"Rank={wgth.get_rank()}, local_rank={wgth.get_local_rank()}")
    global_comm, local_comm = wgth.init_torch_env_and_create_wm_comm(
        wgth.get_rank(),
        wgth.get_world_size(),
        wgth.get_local_rank(),
        wgth.get_local_size(),
        args.distributed_backend_type,
        args.log_level
    )   

    if args.use_cpp_ext:
        wgth.compile_cpp_extension()

    #数据集加载 考虑用torchrec生成的随机数据集
    graph_comm = local_comm
    if global_comm.get_size() != local_comm.get_size() and global_comm.support_type_location("continuous", "cuda"):
        print("Using global communicator for graph structure.")
        if not args.use_global_embedding:
            args.use_global_embedding = True
            print("Changing to using global communicator for embedding...")
            if args.embedding_memory_type == "chunked":
                print("Changing to continuous wholememory for embedding...")
                args.embedding_memory_type = "continuous"

    feature_comm = global_comm if args.use_global_embedding else local_comm
                
    embedding_wholememory_type = args.embedding_memory_type
    embedding_wholememory_location = (
        "cpu" if args.cache_type != "none" or args.cache_ratio == 0.0 else "cuda"
    )
    if args.cache_ratio == 0.0:
        args.cache_type = "none"
    access_type = "readonly" if args.train_embedding is False else "readwrite"
    cache_policy = wgth.create_builtin_cache_policy(
        args.cache_type,
        embedding_wholememory_type,
        embedding_wholememory_location,
        access_type,
        args.cache_ratio,
    )     

    wm_optimizer = None
    embedding_dtype = torch.float32 if not args.fp16_mbedding else torch.float16

    node_feat_wm_embedding = wgth.create_embedding(
        global_comm,
        "distributed",
        "cuda",
        torch.float16,
        [args.node_num, args.feat_dim],
        cache_policy=cache_policy,
        random_init=True,
        round_robin_size=args.round_robin_size,
    )

    wm_optimizer = wgth.create_wholememory_optimizer(node_feat_wm_embedding, "adam", {})
    wgth.set_framework(args.framework)
    # Create GatherFn module
    model = GatherFn(node_feat_wm_embedding).cuda()
    model = DDP(model, delay_allreduce=True)
    # Generate random input data
    input_data = torch.randn(args.node_num).cuda()
    optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=args.lr)
    # Measure forward and backward time
    # forward_time, backward_time = measure_time(model, input_data, args.epochs)
    train(input_data, model, optimizer, wm_optimizer, global_comm)
    # print(f"Forward time: {forward_time:.6f} seconds")
    # print(f"Backward time: {backward_time:.6f} seconds")
    
    wgth.finalize()
if __name__ == "__main__":
    wgth.distributed_launch(args, main)