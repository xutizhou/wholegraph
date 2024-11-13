import torch
from .embedding import WholeMemoryEmbedding, WholeMemoryEmbeddingModule

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