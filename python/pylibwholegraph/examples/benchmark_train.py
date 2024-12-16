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
            min_num_ids = self.min_ids_per_features
            max_num_ids = self.ids_per_features
            length = torch.full((self.batch_size,), self.ids_per_features)
            value = torch.randint(
                0, hash_size, (int(length.sum()),)
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

argparser.add_argument(
    "--batch_size", type=int, default=64, help="Batch size"
)
argparser.add_argument(
    "--node_count", type=int, default=5000000, help="Node count"
)
args = argparser.parse_args()
num_steps = 100
args.epochs = 100
args.feat_dim = 128
seq_len = 4000
def train(model, wm_optimizer, global_comm):
    if wgth.get_rank() == 0:
        print("start training...")
    print(f"world_size={wgth.get_world_size()}, rank={wgth.get_rank()}")

    print(f"args.feat_dim={args.feat_dim}")
    print(f"args.batch_size={args.batch_size}")
    print(f"seq_len={seq_len}")
    print(f"args.epochs={args.epochs}")
    print(f"args.node_count={args.node_count}")
    print(f"num_steps={num_steps}")
    
    dataset = CustomDataset(num_steps,args.node_count,batch_size=args.batch_size, seq_len=seq_len, device=torch.device("cuda"))

   

    #get dataset size
    cnt = 0
    for step in range(num_steps):
        features = dataset.__getitem__(step)
        cnt += features.shape[0]
    print(f"dataset size={cnt}")

    # train model
    train_start_time = time.perf_counter()
    train_step = 0
    epoch = 0  
    while epoch < args.epochs:
        for step in range(num_steps):
            features = dataset.__getitem__(step)
            features = features.to("cuda")            
            # idx = idx.cuda(non_blocking=True)
            model.train()
            # torch.cuda.nvtx.range_push("Forward Pass")
            logits = model(features)  
            # torch.cuda.nvtx.range_pop()         
            loss = torch.cat((logits,), dim=1).sum()
            # torch.cuda.nvtx.range_push("Backward Pass")
            loss.backward()
            # torch.cuda.nvtx.range_pop() 
            # torch.cuda.nvtx.range_push("Update Gradient Pass")
            if wm_optimizer is not None:
                wm_optimizer.step(args.lr * 0.1)
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
            % (train_time / args.epochs,)
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
        embedding_wholememory_type,
        embedding_wholememory_location,
        embedding_dtype,
        [args.node_count, args.feat_dim],
        cache_policy=cache_policy,
        random_init=True,
        round_robin_size=args.round_robin_size,
    )

    wm_optimizer = wgth.create_wholememory_optimizer(node_feat_wm_embedding, "sgd", {})
    print(f"wm_optimizer={wm_optimizer}")
    wgth.set_framework(args.framework)
    # Create GatherFn module
    
    model = GatherFn(node_feat_wm_embedding).cuda()
    model = DDP(model, delay_allreduce=True)
    # Generate random input data
    # optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=args.lr)
    # Measure forward and backward time
    # forward_time, backward_time = measure_time(model, input_data, args.epochs)
    train(model, wm_optimizer, global_comm)
    # print(f"Forward time: {forward_time:.6f} seconds")
    # print(f"Backward time: {backward_time:.6f} seconds")
    
    wgth.finalize()
if __name__ == "__main__":
    wgth.distributed_launch(args, main)