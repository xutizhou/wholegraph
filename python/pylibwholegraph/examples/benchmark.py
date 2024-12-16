import subprocess

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

from pylibwholegraph.torch.embedding import WholeMemoryEmbedding, WholeMemoryEmbeddingModule
import torch
from torch.profiler import profile, record_function, ProfilerActivity

def print_gpu_memory_usage():
    allocated = torch.cuda.memory_allocated() / 1024**2  # 转换为 MB
    reserved = torch.cuda.memory_reserved() / 1024**2  # 转换为 MB
    print(f"Allocated memory: {allocated:.2f} MB")
    print(f"Reserved memory: {reserved:.2f} MB")
    
class CustomDataset():
    def __init__(self, num_steps, hash_size, batch_size, seq_len, device):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.hash_size = hash_size
        self.device = device
        self.min_ids_per_features = 0
        self.ids_per_features = seq_len
        self.keys = ["feature_0"]
        self.data = self._generate_data()
    def _generate_data(self):
        data = []
        for _ in range(self.num_steps):
            values = []
            lengths = []
            hash_size = self.hash_size
            length = torch.full((self.batch_size,), self.ids_per_features, device=self.device)
            value = torch.randint(
                0, hash_size, (int(length.sum()),),device=self.device
            )
            lengths.append(length)
            values.append(value)
            sparse_features = torch.cat(values)
            data.append(sparse_features)
        return data
    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        return self.data[idx]
class GatherFn(torch.nn.Module):
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

batch_size = 2048
node_count=80000000
epochs = 100
dataset_size = 800000000
feat_dim = 128
seq_len = 64
num_steps = dataset_size // (batch_size * seq_len)



print(f"node_count: {node_count}")
print(f'node_count GB: {node_count * feat_dim * 4 / 1024 / 1024 / 1024}')
print(f"embedding_dim: {feat_dim}")
print(f"batch_size: {batch_size}")
print(f"seq_len: {seq_len}")
print(f"dataset_size: {dataset_size}")
print(f"fetched embedding size GB: {dataset_size * feat_dim * 4 / 1024 / 1024 / 1024}")
print(f"num_epochs: {epochs}")
print(f"num_steps: {num_steps}")

def train(model, wm_optimizer, global_comm):
    if wgth.get_rank() == 0:
        print("start training...")
    print(f"world_size={wgth.get_world_size()}, rank={wgth.get_rank()}")

    print(f"feat_dim={feat_dim}")
    print(f"batch_size={batch_size}")
    print(f"seq_len={seq_len}")
    print(f"epochs={epochs}")
    print(f"node_count={node_count}")
    print(f"num_steps={num_steps}")

    # # Run nvidia-smi command and capture its output
    # result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    # print(result.stdout.decode())
    dataset = CustomDataset(num_steps,node_count,batch_size=batch_size, seq_len=seq_len, device=torch.device("cuda"))
    # train_dataloader = wgth.get_train_dataloader(
    #     dataset,
    #     1,
    #     replica_id=wgth.get_rank(),
    #     num_replicas=wgth.get_world_size(),
    #     num_workers=args.dataloaderworkers,
    # )
    # result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    # print(result.stdout.decode())    

    #get dataset size
    cnt = 0
    for _, features in enumerate(dataset):
        if _==0:
            print(f"features.shape={features.shape}")
            print(f"features={features}")
        cnt += features.shape[0]
    print(f"dataset size={cnt}")

    # train model
    train_start_time = time.perf_counter()
    train_step = 0
    epoch = 0  
    while epoch < epochs:
        for step, features in enumerate(dataset):
            features = torch.squeeze(features).to("cuda")
            # print(f"features at rank:{features.get_device()}current GPU is{wgth.get_rank()}")          
            # idx = idx.cuda(non_blocking=True)
            model.train()
            # torch.cuda.nvtx.range_push("Forward Pass")
            logits = model(features)  
            # if step == 0:
            #     print(f"logits.shape={logits.shape}")
            #     print(f"logits={logits}")

            # torch.cuda.nvtx.range_pop() 
            # if wgth.get_rank() == 0 and train_step % 100 == 0:
            #     print(
            #         "[%s] [LOSS] step=%d, loss=%f"
            #         % (
            #             datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            #             train_step,
            #             loss.cpu().item(),
            #         )
            #     )
            train_step = train_step + 1
        epoch = epoch + 1
    global_comm.barrier()
    train_end_time = time.perf_counter()
    train_time = train_end_time - train_start_time
    if wgth.get_rank() == 0:
        print(
            "[%s] [TRAIN_TIME] train time is %.2f seconds"
            % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), train_time)
        )
        print(
            "[EPOCH_TIME] %.2f seconds."
            % (train_time / epochs,)
        )

def main():

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
        embedding_wholememory_type,
        embedding_wholememory_location,
        embedding_dtype,
        [node_count, feat_dim],
        cache_policy=cache_policy,
        random_init=True,
        round_robin_size=args.round_robin_size,
    )
    print(f"gpu utilization after create_embedding:{wgth.get_rank()}")
    print_gpu_memory_usage()
    wm_optimizer = wgth.create_wholememory_optimizer(node_feat_wm_embedding, "sgd", {})
    print(f"wm_optimizer={wm_optimizer}")
    wgth.set_framework(args.framework)
    # Create GatherFn module
    
    model = GatherFn(node_feat_wm_embedding)
    model.cuda()
    model = DDP(model, delay_allreduce=True)
    # Generate random input data
    # optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=args.lr)
    # Measure forward and backward time
    # forward_time, backward_time = measure_time(model, input_data, epochs)
    train(model, wm_optimizer, global_comm)
    # print(f"Forward time: {forward_time:.6f} seconds")
    # print(f"Backward time: {backward_time:.6f} seconds")
    
    wgth.finalize()
if __name__ == "__main__":
    wgth.distributed_launch(args, main)