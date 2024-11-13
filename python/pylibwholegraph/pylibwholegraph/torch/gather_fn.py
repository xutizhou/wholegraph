import torch.nn as nn

class GatherFn(nn.Module):
    def __init__(self, node_feat_wm_embedding):
        super(GatherFn, self).__init__()
        self.node_feat_wm_embedding = node_feat_wm_embedding

    def forward(self, x):
        # 定义前向传播逻辑
        pass