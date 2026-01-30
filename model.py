import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class FastPoseViT(nn.Module):
    def __init__(self, model_name='google/vit-base-patch16-224', pretrained=True):
        super(FastPoseViT, self).__init__()
        if pretrained:
            self.vit = ViTModel.from_pretrained(model_name)
        else:
            config = ViTConfig.from_pretrained(model_name)
            self.vit = ViTModel(config)
        embed_dim = self.vit.config.hidden_size
        self.pose_head = nn.Sequential(
            nn.Linear(embed_dim, 9),
            )
        nn.init.xavier_uniform_(self.pose_head[0].weight)
        nn.init.zeros_(self.pose_head[0].bias)

    def forward(self, x):
        
        outputs = self.vit(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0, :] 
        pose = self.pose_head(cls_token)
        
        return {
            'U': pose[:, :3],
            'R6D': pose[:, 3:]
        }