import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class NonLinear_Attention(nn.Module):#soft attentionの実装
    '''
    アテンション機構 (Attention mechanism)
    dim_feature  : 出力の特徴次元
    dim_attention: アテンション機構の次元
    '''
    def __init__(self, dim_feature: int,  dim_attention: int):
        super().__init__()

        # z: 特徴量集合H(画像を分割した特徴量)の各要素を変換する全結合層(Wz)
        self.att_for_Keys = nn.Linear(dim_feature, dim_attention)#baseはdim_encoder=2048, depthではMLPの場合dim_encoder=2048+mlp_dim_out

        # h: 1つの特徴量h出力を変換する全結合層(Wh)
        self.att_for_Ques = nn.Linear(dim_feature, dim_attention)

        # e: アライメントスコアを計算するための全結合層
        self.full_att = nn.Linear(dim_attention, 1)

        # α: アテンション重みを計算する活性化関数
        self.relu = nn.ReLU(inplace=True)

    '''
    アテンション機構の順伝播
    encoder_out   : エンコーダ出力,
                    [バッチサイズ, 特徴マップの幅 * 高さ, チャネル数]
    decoder_hidden: デコーダ隠れ状態の次元
    '''
    def forward(self, Key_features: torch.Tensor, 
                Que_features: torch.Tensor):
        
        # 並び替え
        # -> [バッチサイズ, 特徴マップの幅 * 高さ, チャネル数]
        Key_features = Key_features.permute(0, 2, 3, 1).flatten(1, 2)#[バッチサイズ, 196, チャンネル数]
        Que_features = Que_features.permute(0, 2, 3, 1).flatten(1, 2)
        # e: アライメントスコア
        Keys = self.att_for_Keys(Key_features)    # Wz * z [バッチサイズ, 196, D]=>[バッチサイズ, 196, att_D]
        Ques = self.att_for_Ques(Que_features) # Wh * h_{t-1} [バッチサイズ, hidden_D]=>[バッチサイズ, att_D]
        att = self.full_att(self.relu(Keys.unsqueeze(1)+Ques.unsqueeze(2))).squeeze(3)#[バッチサイズ, 196, 196]

        # α: T個の部分領域ごとのアテンション重み
        alpha = att.softmax(dim=2) #[バッチサイズ, 196, 196]

        # c: コンテキストベクトル
        #context_vector = (Key_features * alpha.unsqueeze(2)).sum(dim=1)#[バッチサイズ, 196, D]=>[バッチサイズ, D]
        context_vector = torch.bmm(alpha, Key_features)#[バッチサイズ, 196, チャンネル数]

        return context_vector, alpha
    
    
    '''
    apply attention on samples and detect false neagtive samples
    '''
    @torch.no_grad()
    def samples_attention(self, features: torch.Tensor, batch_size: int, n_views: int, device: str):
        
        att1 = self.att_for_set(features) # [バッチサイズ, D] => [バッチサイズ, att_D]
        att2 = self.att_for_one(features) # [バッチサイズ, D] => [バッチサイズ, att_D]

        att1 = att1.unsqueeze(0) # [1, バッチサイズ, att_D]
        att2 = att2.unsqueeze(1) # [バッチサイズ, 1, att_D]

        att = self.full_att(self.relu(att1+att2)).squeeze(2) # [バッチサイズ, バッチサイズ, att_D]=>[バッチサイズ, バッチサイズ, 1]=>[バッチサイズ, バッチサイズ]

        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)#[0,1,2,....,0,1,2]
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()#[2batch_size,2batch_size]:(i,j)がペアなら1 else 0
        labels = labels.to(device)

        att = att[~labels.bool()].view(att.shape[0], -1) #[2batch_size-2, 2batch_size-2] 自分と正例ペアのスコアを除外

        alpha = att.softmax(dim=1)

        return alpha
    

class Dot_Attention(nn.Module):#soft attentionの実装
    '''
    アテンション機構 (Attention mechanism)
    dim_feature  : 出力の特徴次元
    dim_attention: アテンション機構の次元
    '''
    def __init__(self, dim_feature: int,  dim_attention: int):
        super().__init__()

        # z: 特徴量集合H(画像を分割した特徴量)の各要素を変換する全結合層(Wz)
        self.att_for_Keys = nn.Linear(dim_feature, dim_attention)#baseはdim_encoder=2048, depthではMLPの場合dim_encoder=2048+mlp_dim_out

        # h: 1つの特徴量h出力を変換する全結合層(Wh)
        self.att_for_Ques = nn.Linear(dim_feature, dim_attention)

        # e: アライメントスコアを計算するための全結合層
        # self.full_att = nn.Linear(dim_attention, 1)

        # α: アテンション重みを計算する活性化関数
        # self.relu = nn.ReLU(inplace=True)

    '''
    アテンション機構の順伝播
    encoder_out   : エンコーダ出力,
                    [バッチサイズ, 特徴マップの幅 * 高さ, チャネル数]
    decoder_hidden: デコーダ隠れ状態の次元
    '''
    def forward(self, Key_features: torch.Tensor, 
                Que_features: torch.Tensor):
        
        # 並び替え
        # -> [バッチサイズ, 特徴マップの幅 * 高さ, チャネル数]
        Key_features = Key_features.permute(0, 2, 3, 1).flatten(1, 2)#[バッチサイズ, 196, チャンネル数]
        Que_features = Que_features.permute(0, 2, 3, 1).flatten(1, 2)
        # e: アライメントスコア
        Keys = self.att_for_Keys(Key_features)    # Wz * z [バッチサイズ, 196, D]=>[バッチサイズ, 196, att_D]
        Ques = self.att_for_Ques(Que_features) # Wh * h_{t-1} [バッチサイズ, hidden_D]=>[バッチサイズ, att_D]
        d_k = Keys.size(-1)
        att = torch.matmul(Ques, Keys.transpose(-2, -1))/(d_k**0.5)# [バッチサイズ, 196, 196]
        #att = self.full_att(self.relu(Keys.unsqueeze(1)+Ques.unsqueeze(2))).squeeze(3)#[バッチサイズ, 196, 196]

        # α: T個の部分領域ごとのアテンション重み
        alpha = att.softmax(dim=-1) #[バッチサイズ, 196, 196]

        # c: コンテキストベクトル
        #context_vector = (Key_features * alpha.unsqueeze(2)).sum(dim=1)#[バッチサイズ, 196, D]=>[バッチサイズ, D]
        context_vector = torch.bmm(alpha, Key_features)#[バッチサイズ, 196, チャンネル数]

        return context_vector, alpha
    
    
    '''
    apply attention on samples and detect false neagtive samples
    '''
    @torch.no_grad()
    def samples_attention(self, features: torch.Tensor, batch_size: int, n_views: int, device: str):
        
        att1 = self.att_for_set(features) # [バッチサイズ, D] => [バッチサイズ, att_D]
        att2 = self.att_for_one(features) # [バッチサイズ, D] => [バッチサイズ, att_D]

        att1 = att1.unsqueeze(0) # [1, バッチサイズ, att_D]
        att2 = att2.unsqueeze(1) # [バッチサイズ, 1, att_D]

        att = self.full_att(self.relu(att1+att2)).squeeze(2) # [バッチサイズ, バッチサイズ, att_D]=>[バッチサイズ, バッチサイズ, 1]=>[バッチサイズ, バッチサイズ]

        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)#[0,1,2,....,0,1,2]
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()#[2batch_size,2batch_size]:(i,j)がペアなら1 else 0
        labels = labels.to(device)

        att = att[~labels.bool()].view(att.shape[0], -1) #[2batch_size-2, 2batch_size-2] 自分と正例ペアのスコアを除外

        alpha = att.softmax(dim=1)

        return alpha