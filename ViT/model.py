import torch
import torch.nn as nn

class LinearProjection(nn.Module):

    def __init__(self, patch_vec_size, num_patches, latent_vec_dim, drop_rate):
        super().__init__()
        self.linear_proj = nn.Linear(patch_vec_size, latent_vec_dim) # (1 * p^2 c) -> (1 x D)
        self.cls_token = nn.Parameter(torch.randn(1, latent_vec_dim)) # (1xD)의 클래스 토큰을 가장 왼쪽에 concat. 시켜줘야 하기 때문에 우선 (1xD) 사이즈의 학습 가능한 파라미터로 정의.
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, latent_vec_dim))
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x): # 배치 단위로 (b x N x p^2 c) 들어온다.
        batch_size = x.size(0)
        x = torch.cat([self.cls_token.repeat(batch_size, 1, 1), self.linear_proj(x)], dim=1) 
        # linear_proj 이후엔 (b x N x D), repeat 함수를 통해 cls_token의 사이즈를 (b x 1 x D)로 조정.
        x += self.pos_embedding
        x = self.dropout(x)
        return x # Transformer의 Input을 return. 

class MultiheadedSelfAttention(nn.Module):
    def __init__(self, latent_vec_dim, num_heads, drop_rate):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_heads = num_heads
        self.latent_vec_dim = latent_vec_dim
        self.head_dim = int(latent_vec_dim / num_heads)
        self.query = nn.Linear(latent_vec_dim, latent_vec_dim) # D -> D_h, D = k x D_h, 즉 D_h를 헤드 수만큼 계산 했다.
        self.key = nn.Linear(latent_vec_dim, latent_vec_dim)
        self.value = nn.Linear(latent_vec_dim, latent_vec_dim)
        self.scale = torch.sqrt(self.head_dim * torch.ones(1)).to(device)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        batch_size = x.size(0)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (b x N x h x D_h) -> (b x h x N x D_h), 헤드 수에 따라 qkv 값이 달라지니까 permute로 조정.
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 3, 1) # k^Transpose
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attention = torch.softmax(q @ k / self.scale, dim = -1)
        x = self.dropout(attention) @ v
        x = x.permute(0,2,1,3).reshape(batch_size, -1, self.latent_vec_dim)

        return x, attention

class TFencoderLayer(nn.Module):
    def __init__(self, latent_vec_dim, num_heads, mlp_hidden_dim, drop_rate):
        super().__init__()
        self.ln1 = nn.LayerNorm(latent_vec_dim)
        self.ln2 = nn.LayerNorm(latent_vec_dim)
        self.msa = MultiheadedSelfAttention(latent_vec_dim = latent_vec_dim, num_heads = num_heads, drop_rate = drop_rate)
        self.dropout = nn.Dropout(drop_rate)
        self.mlp = nn.Sequential(nn.Linear(latent_vec_dim, mlp_hidden_dim),
                                 nn.GELU(), nn.Dropout(drop_rate),
                                 nn.Linear(mlp_hidden_dim, latent_vec_dim),
                                 nn.Dropout(drop_rate))

    def forward(self, x):
        z = self.ln1(x)
        z, att = self.msa(z)
        z = self.dropout(z)
        x = x + z
        z = self.ln2(x)
        z = self.mlp(z)
        x = x + z

        return x, att

class VisionTransformer(nn.Module):
    def __init__(self, patch_vec_size, num_patches, latent_vec_dim, num_heads, mlp_hidden_dim, drop_rate, num_layers, num_classes):
        super().__init__()
        self.patchembedding = LinearProjection(patch_vec_size = patch_vec_size, num_patches = num_patches, # patch_vec_size = p^2 * c
                                               latent_vec_dim = latent_vec_dim, drop_rate = drop_rate)
        self.transformer = nn.ModuleList([TFencoderLayer(latent_vec_dim = latent_vec_dim, num_heads = num_heads,
                                                         mlp_hidden_dim = mlp_hidden_dim, drop_rate = drop_rate)
                                          for _ in range(num_layers)]) # Transformer의 Layer가 반복되는 과정을 리스트 안에 넣어서 처리. num_layer만큼 list에 append 한다.

        self.mlp_head = nn.Sequential(nn.LayerNorm(latent_vec_dim), nn.Linear(latent_vec_dim, num_classes))

    def forward(self, x):
        att_list = []
        x = self.patchembedding(x)
        for layer in self.transformer: # List로 구성되어 있는 레이어를 하나씩 불러
            x, att = layer(x)
            att_list.append(att)
        x = self.mlp_head(x[:,0]) # class token에 해당되는 부분의 벡터만 사용

        return x, att_list
