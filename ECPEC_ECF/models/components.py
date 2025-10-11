# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class BiLSTMEncoder(nn.Module):
    """双向LSTM编码器"""

    def __init__(self, input_dim, hidden_dim, dropout=0.1, num_layers=1):
        super(BiLSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        # 只有在多层时才使用dropout
        lstm_dropout = dropout if num_layers > 1 else 0
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, bidirectional=True, dropout=lstm_dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            lengths: (batch_size,) 实际序列长度
        Returns:
            output: (batch_size, seq_len, hidden_dim * 2)
        """
        if lengths is not None:
            # 按长度打包序列以提高效率
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output, (h_n, c_n) = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, (h_n, c_n) = self.lstm(x)

        return self.dropout(output)


class AttentionLayer(nn.Module):
    """注意力层"""

    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(hidden_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, hidden_dim)
            mask: (batch_size, seq_len) 1表示有效位置，0表示padding
        Returns:
            attended: (batch_size, hidden_dim)
            attention_weights: (batch_size, seq_len)
        """
        # 计算注意力权重
        u = self.tanh(self.w1(x))  # (batch_size, seq_len, hidden_dim)
        scores = self.w2(u).squeeze(-1)  # (batch_size, seq_len)

        # 应用mask
        if mask is not None:
            # 确保mask和scores的维度匹配
            seq_len = scores.size(-1)
            if mask.size(-1) != seq_len:
                # 如果mask长度不匹配，截断或填充
                if mask.size(-1) > seq_len:
                    mask = mask[:, :seq_len]
                else:
                    # 用0填充（表示padding位置）
                    batch_size = mask.size(0)
                    padded_mask = torch.zeros(batch_size, seq_len, device=mask.device, dtype=mask.dtype)
                    padded_mask[:, :mask.size(-1)] = mask
                    mask = padded_mask

            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = self.softmax(scores)  # (batch_size, seq_len)

        # 加权求和
        attended = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)  # (batch_size, hidden_dim)

        return attended, attention_weights


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)

        # 线性变换并重塑为多头
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 应用注意力权重
        attended = torch.matmul(attention_weights, V)

        # 重新整合多头
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        return self.w_o(attended)


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    """Transformer块"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # 自注意力
        attended = self.attention(x, x, x, mask)
        x = self.norm1(x + attended)

        # 前馈网络
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x


class BERTEncoder(nn.Module):
    """BERT编码器封装"""

    def __init__(self, model_name='../roberta', hidden_dim=768,
                 dropout=0.1, freeze_bert=False):
        super(BERTEncoder, self).__init__()

        model_name = ensure_model_path_exists(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_dim = hidden_dim

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # 如果需要调整维度
        if self.bert.config.hidden_size != hidden_dim:
            self.projection = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        else:
            self.projection = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None, output_type='sequence'):
        """
        Args:
            input_ids: (batch_size, seq_len) 或 (batch_size, doc_len, seq_len)
            attention_mask: 对应的attention mask
            output_type: 'sequence' 或 'pooled'
        """
        original_shape = input_ids.shape

        # 如果是3D输入，重塑为2D
        if len(original_shape) == 3:
            batch_size, doc_len, seq_len = original_shape
            input_ids = input_ids.view(-1, seq_len)
            if attention_mask is not None:
                attention_mask = attention_mask.view(-1, seq_len)

        # BERT编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if output_type == 'pooled':
            encoded = outputs.pooler_output  # (batch_size * doc_len, hidden_size)
        else:
            encoded = outputs.last_hidden_state  # (batch_size * doc_len, seq_len, hidden_size)

        # 投影到目标维度
        if self.projection is not None:
            encoded = self.projection(encoded)

        encoded = self.dropout(encoded)

        # 恢复原始形状
        if len(original_shape) == 3:
            if output_type == 'pooled':
                encoded = encoded.view(batch_size, doc_len, -1)
            else:
                encoded = encoded.view(batch_size, doc_len, seq_len, -1)

        return encoded


class BERTDocEncoder(nn.Module):
    """BERT_doc编码器：拼接整个对话为长序列，按索引提取各话语CLS token"""

    def __init__(self, model_name='../roberta', hidden_dim=768,
                 dropout=0.1, freeze_bert=False, max_doc_len=35, max_sen_len=35):
        super(BERTDocEncoder, self).__init__()

        model_name = ensure_model_path_exists(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hidden_dim = hidden_dim
        self.max_doc_len = max_doc_len
        self.max_sen_len = max_sen_len

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # 如果需要调整维度
        if self.bert.config.hidden_size != hidden_dim:
            self.projection = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        else:
            self.projection = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, texts, doc_lengths):
        """
        Args:
            texts: List[List[str]] - batch中每个对话的话语文本列表
            doc_lengths: (batch_size,) - 每个对话的实际长度
        Returns:
            encoded: (batch_size, max_doc_len, hidden_dim) - 每个话语的CLS表示
        """
        batch_size = len(texts)
        device = next(self.bert.parameters()).device

        # 存储结果
        batch_utterance_features = []

        for batch_idx in range(batch_size):
            doc_texts = texts[batch_idx]
            actual_doc_len = doc_lengths[batch_idx].item()

            # 构建完整对话序列：为每个话语显式插入 [CLS] 作为话语起始标记，再接 [SEP]
            # 格式：[CLS] utt1 [SEP] [CLS] utt2 [SEP] ... [CLS] uttN [SEP]
            full_tokens = []
            utterance_start_positions = []  # 记录每个话语 [CLS] 的位置

            for utt_idx in range(actual_doc_len):
                if utt_idx < len(doc_texts):
                    utt_text = doc_texts[utt_idx].strip()
                else:
                    utt_text = ""  # padding话语

                # 为该话语添加 [CLS] 并记录其位置
                full_tokens.append(self.tokenizer.cls_token)
                utterance_start_positions.append(len(full_tokens) - 1)

                # 添加话语tokens（不包括特殊token）
                if utt_text:
                    utt_tokens = self.tokenizer.tokenize(utt_text)
                    full_tokens.extend(utt_tokens)

                # 话语结束添加 [SEP]
                full_tokens.append(self.tokenizer.sep_token)

            # 转换为ID并处理长度
            token_ids = self.tokenizer.convert_tokens_to_ids(full_tokens)
            max_length = self.max_doc_len * self.max_sen_len

            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                # 调整超出范围的位置
                utterance_start_positions = [pos for pos in utterance_start_positions if pos < max_length]

            # 创建attention mask和padding
            attention_mask = [1] * len(token_ids)
            while len(token_ids) < max_length:
                token_ids.append(self.tokenizer.pad_token_id)
                attention_mask.append(0)

            input_ids = torch.tensor([token_ids], device=device)
            attention_mask_tensor = torch.tensor([attention_mask], device=device)

            # BERT编码
            with torch.set_grad_enabled(self.training):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask_tensor)
                sequence_output = outputs.last_hidden_state  # (1, seq_len, hidden_size)

            # 提取各话语的表示
            utterance_features = []
            for utt_idx in range(self.max_doc_len):
                if utt_idx < actual_doc_len and utt_idx < len(utterance_start_positions):
                    # 提取对应位置的token表示
                    pos = utterance_start_positions[utt_idx]
                    if pos < sequence_output.size(1):
                        utt_feat = sequence_output[0, pos, :]  # (hidden_size,)
                    else:
                        # 位置超出序列长度，使用零向量
                        utt_feat = torch.zeros(sequence_output.size(-1), device=device)
                else:
                    # padding话语，使用零向量
                    utt_feat = torch.zeros(sequence_output.size(-1), device=device)

                utterance_features.append(utt_feat)

            # 堆叠为 (max_doc_len, hidden_size)
            doc_features = torch.stack(utterance_features, dim=0)
            batch_utterance_features.append(doc_features)

        # 堆叠为 (batch_size, max_doc_len, hidden_size)
        encoded = torch.stack(batch_utterance_features, dim=0)

        # 投影到目标维度
        if self.projection is not None:
            encoded = self.projection(encoded)

        encoded = self.dropout(encoded)

        return encoded



class IdentityTextEncoder(nn.Module):
    """恒等文本编码器：直接返回预计算特征。
    期望输入形状：(batch_size, max_doc_len, hidden_dim)
    """
    def __init__(self):
        super().__init__()

    def forward(self, features, *args, **kwargs):
        return features


class MultimodalFusion(nn.Module):
    """多模态特征融合"""

    def __init__(self, text_dim, audio_dim, visual_dim, hidden_dim,
                 fusion_type='concat', dropout=0.1):
        super(MultimodalFusion, self).__init__()

        self.fusion_type = fusion_type
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim

        # 模态特异性投影
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)

        # 融合策略
        if fusion_type == 'concat':
            self.fusion_layer = nn.Linear(hidden_dim * 3, hidden_dim)
        elif fusion_type == 'attention':
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout)
        elif fusion_type == 'gate':
            self.gate = nn.Linear(hidden_dim * 3, 3)

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_feat, audio_feat, visual_feat):
        """
        Args:
            text_feat: (batch_size, seq_len, text_dim) 或 (batch_size, text_dim)
            audio_feat: (batch_size, seq_len, audio_dim) 或 (batch_size, audio_dim)
            visual_feat: (batch_size, seq_len, visual_dim) 或 (batch_size, visual_dim)
        """
        # 投影到统一维度
        text_proj = self.text_proj(text_feat)
        audio_proj = self.audio_proj(audio_feat)
        visual_proj = self.visual_proj(visual_feat)

        if self.fusion_type == 'concat':
            # 简单拼接
            fused = torch.cat([text_proj, audio_proj, visual_proj], dim=-1)
            fused = self.fusion_layer(fused)

        elif self.fusion_type == 'attention':
            # 注意力融合
            modalities = torch.stack([text_proj, audio_proj, visual_proj], dim=-2)  # (..., 3, hidden_dim)
            if len(modalities.shape) == 3:  # (batch_size, 3, hidden_dim)
                modalities = modalities.transpose(0, 1)  # (3, batch_size, hidden_dim)
                fused, _ = self.attention(modalities, modalities, modalities)
                fused = fused.mean(dim=0)  # (batch_size, hidden_dim)
            else:  # (batch_size, seq_len, 3, hidden_dim)
                batch_size, seq_len = modalities.shape[:2]
                modalities = modalities.view(-1, 3, self.hidden_dim).transpose(0, 1)
                fused, _ = self.attention(modalities, modalities, modalities)
                fused = fused.mean(dim=0).view(batch_size, seq_len, self.hidden_dim)

        elif self.fusion_type == 'gate':
            # 门控融合
            concat_feat = torch.cat([text_proj, audio_proj, visual_proj], dim=-1)
            gate_weights = F.softmax(self.gate(concat_feat), dim=-1)

            modalities = torch.stack([text_proj, audio_proj, visual_proj], dim=-1)
            fused = torch.sum(modalities * gate_weights.unsqueeze(-2), dim=-1)

        else:
            # 默认简单平均
            fused = (text_proj + audio_proj + visual_proj) / 3

        fused = self.layer_norm(fused)
        fused = self.dropout(fused)

        return fused


class FeatureProjection(nn.Module):
    """特征投影层"""

    def __init__(self, input_dim, output_dim, activation='relu', dropout=0.1):
        super(FeatureProjection, self).__init__()

        self.projection = nn.Linear(input_dim, output_dim)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = None

        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.projection(x)
        if self.activation is not None:
            x = self.activation(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class MaskGenerator(nn.Module):
    """掩码生成器"""

    @staticmethod
    def create_padding_mask(lengths, max_len):
        """创建padding掩码"""
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) < lengths.unsqueeze(1)
        return mask

    @staticmethod
    def create_causal_mask(seq_len):
        """创建因果掩码（下三角矩阵）"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.bool()


class GraphAdjacencyBuilder:
    """图邻接矩阵构建器"""

    @staticmethod
    def build_adjacency(speakers, doc_lengths, max_doc_len, window_size=3,
                       use_speaker_edges=True, use_temporal_edges=True,
                       distance_decay=True, tau=2.0):
        """
        构建对话图的邻接矩阵

        Args:
            speakers: (batch_size, max_doc_len) 说话人ID
            doc_lengths: (batch_size,) 实际对话长度
            max_doc_len: 最大对话长度
            window_size: 时间窗口大小
            use_speaker_edges: 是否使用说话人边
            use_temporal_edges: 是否使用时间边
            distance_decay: 是否使用距离衰减权重
            tau: 距离衰减参数

        Returns:
            adjacency: (batch_size, max_doc_len, max_doc_len) 邻接矩阵
            edge_weights: (batch_size, max_doc_len, max_doc_len) 边权重
        """
        batch_size = speakers.size(0)
        device = speakers.device
        L = max_doc_len

        # 初始化邻接矩阵和权重矩阵
        adjacency = torch.zeros(batch_size, L, L, device=device)
        edge_weights = torch.zeros(batch_size, L, L, device=device)

        # 有效位置掩码（按对话实际长度）
        idx = torch.arange(L, device=device)
        valid = idx.unsqueeze(0) < doc_lengths.unsqueeze(1)  # (B, L)
        pair_valid = valid.unsqueeze(1) & valid.unsqueeze(2)  # (B, L, L)

        # 自环：仅在有效对角处
        I = torch.eye(L, device=device)
        adjacency = adjacency + (I.unsqueeze(0) * pair_valid.float())
        edge_weights = edge_weights + (I.unsqueeze(0) * pair_valid.float())

        # 距离矩阵（用于时间边与说话人边权重）
        dist = torch.abs(idx.view(1, L) - idx.view(L, 1))  # (L, L)
        dist_b = dist.unsqueeze(0).expand(batch_size, -1, -1)  # (B, L, L)

        # 时间邻接（非对角，窗口内）
        if use_temporal_edges:
            temporal_mask = (dist_b > 0) & (dist_b <= window_size) & pair_valid  # (B, L, L)
            if distance_decay:
                temp_w = torch.exp(-dist_b.float() / float(tau))
            else:
                temp_w = torch.ones_like(dist_b, dtype=torch.float32)
            adjacency[temporal_mask] = 1.0
            edge_weights[temporal_mask] = temp_w[temporal_mask]

        # 说话人邻接（非对角，窗口内，同说话人），后写入可覆盖时间边权重（与原实现一致）
        if use_speaker_edges:
            same_speaker = speakers.unsqueeze(2).eq(speakers.unsqueeze(1))  # (B, L, L)
            speaker_mask = (dist_b > 0) & (dist_b <= window_size) & same_speaker & pair_valid
            if distance_decay:
                sp_w = torch.exp(-dist_b.float() / float(tau)) + 0.5
            else:
                sp_w = torch.full_like(dist_b, 1.5, dtype=torch.float32)
            adjacency[speaker_mask] = 1.0
            edge_weights[speaker_mask] = sp_w[speaker_mask]

        return adjacency, edge_weights


class GATv2Layer(nn.Module):
    """GATv2注意力层"""

    def __init__(self, in_dim, out_dim, num_heads=8, dropout=0.1,
                 use_edge_weights=True, use_bias=True):
        super(GATv2Layer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.use_edge_weights = use_edge_weights

        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        # 线性投影
        self.w_q = nn.Linear(in_dim, out_dim, bias=False)
        self.w_k = nn.Linear(in_dim, out_dim, bias=False)
        self.w_v = nn.Linear(in_dim, out_dim, bias=False)

        # GATv2的关键：共享注意力机制
        self.attention = nn.Linear(2 * self.head_dim, 1, bias=use_bias)

        # 输出投影
        self.w_o = nn.Linear(out_dim, out_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
        nn.init.xavier_uniform_(self.attention.weight)
        if self.attention.bias is not None:
            nn.init.zeros_(self.attention.bias)

    def forward(self, x, adjacency, edge_weights=None, mask=None):
        """
        Args:
            x: (batch_size, seq_len, in_dim) 节点特征
            adjacency: (batch_size, seq_len, seq_len) 邻接矩阵
            edge_weights: (batch_size, seq_len, seq_len) 边权重
            mask: (batch_size, seq_len) padding掩码

        Returns:
            output: (batch_size, seq_len, out_dim) 更新后的节点特征
        """
        batch_size, seq_len, _ = x.shape

        # 线性投影
        Q = self.w_q(x)  # (B, L, out_dim)
        K = self.w_k(x)  # (B, L, out_dim)
        V = self.w_v(x)  # (B, L, out_dim)

        # 重塑为多头 (B, H, L, d)
        Qh = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        Kh = K.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        Vh = V.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 向量化的 GATv2 注意力分数计算：
        # self.attention 等价于对 [q, k] 做线性：w_q·q + w_k·k + b
        w = self.attention.weight  # (1, 2*d)
        b = self.attention.bias    # (1,) 或 None
        w_q = w[:, :self.head_dim]
        w_k = w[:, self.head_dim:]

        # s_q: (B, H, L, 1); s_k: (B, H, L, 1)
        s_q = F.linear(Qh, w_q, bias=None)
        s_k = F.linear(Kh, w_k, bias=None)

        # 广播到 (B, H, L, L)
        attention_scores = s_q.squeeze(-1).unsqueeze(-1) + s_k.squeeze(-1).unsqueeze(-2)
        if b is not None:
            attention_scores = attention_scores + b

        # 应用邻接掩码
        attention_scores = attention_scores.masked_fill(adjacency.unsqueeze(1) == 0, -1e9)

        # 应用padding掩码
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, L)
            attention_scores = attention_scores.masked_fill(mask_expanded == 0, -1e9)
            mask_expanded_2 = mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, L, 1)
            attention_scores = attention_scores.masked_fill(mask_expanded_2 == 0, -1e9)

        # Softmax归一化
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)

        # 应用边权重
        if self.use_edge_weights and edge_weights is not None:
            attention_weights = attention_weights * edge_weights.unsqueeze(1)

        # 聚合消息 (B, H, L, d)
        aggregated = torch.matmul(attention_weights, Vh)

        # 合并多头 -> (B, L, out_dim)
        output = aggregated.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.out_dim)

        # 输出投影
        output = self.w_o(output)
        output = self.dropout(output)

        return output


class GraphConversationEncoder(nn.Module):
    """基于图神经网络的对话编码器"""

    def __init__(self, hidden_dim, num_layers=2, num_heads=8, dropout=0.1,
                 window_size=3, use_speaker_edges=True, use_temporal_edges=True,
                 distance_decay=True, tau=2.0, use_residual=True):
        super(GraphConversationEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.window_size = window_size
        self.use_speaker_edges = use_speaker_edges
        self.use_temporal_edges = use_temporal_edges
        self.distance_decay = distance_decay
        self.tau = tau
        self.use_residual = use_residual

        # GATv2层
        self.gat_layers = nn.ModuleList([
            GATv2Layer(hidden_dim, hidden_dim, num_heads, dropout, use_edge_weights=True)
            for _ in range(num_layers)
        ])

        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # 前馈网络
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])

        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # 图邻接构建器
        self.adjacency_builder = GraphAdjacencyBuilder()

    def forward(self, x, speakers, doc_lengths, mask=None):
        """
        Args:
            x: (batch_size, max_doc_len, hidden_dim) 节点特征
            speakers: (batch_size, max_doc_len) 说话人ID
            doc_lengths: (batch_size,) 实际对话长度
            mask: (batch_size, max_doc_len) padding掩码

        Returns:
            output: (batch_size, max_doc_len, hidden_dim) 更新后的节点特征
        """
        batch_size, max_doc_len, _ = x.shape

        # 构建图邻接矩阵
        adjacency, edge_weights = self.adjacency_builder.build_adjacency(
            speakers, doc_lengths, max_doc_len,
            window_size=self.window_size,
            use_speaker_edges=self.use_speaker_edges,
            use_temporal_edges=self.use_temporal_edges,
            distance_decay=self.distance_decay,
            tau=self.tau
        )

        # 多层图传播
        output = x
        for i in range(self.num_layers):
            # GATv2层
            gat_output = self.gat_layers[i](output, adjacency, edge_weights, mask)

            # 残差连接和层归一化
            if self.use_residual:
                output = self.layer_norms[i](output + gat_output)
            else:
                output = self.layer_norms[i](gat_output)

            # 前馈网络
            ffn_output = self.ffns[i](output)

            # 残差连接和层归一化
            if self.use_residual:
                output = self.ffn_norms[i](output + ffn_output)
            else:
                output = self.ffn_norms[i](ffn_output)

        return output


if __name__ == "__main__":
    # 测试各个组件
    batch_size, seq_len, hidden_dim = 4, 10, 256

    # 测试BiLSTM
    print("测试BiLSTM编码器...")
    x = torch.randn(batch_size, seq_len, 300)
    lengths = torch.tensor([10, 8, 6, 9])

    bilstm = BiLSTMEncoder(300, 128)
    output = bilstm(x, lengths)
    print(f"BiLSTM输出形状: {output.shape}")

    # 测试注意力
    print("\n测试注意力层...")
    attention = AttentionLayer(256)
    mask = MaskGenerator.create_padding_mask(lengths, seq_len)
    attended, weights = attention(output, mask)
    print(f"注意力输出形状: {attended.shape}")
    print(f"注意力权重形状: {weights.shape}")


    # 测试图邻接构建
    print("\n测试图邻接构建...")
    speakers = torch.tensor([[0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 1, 1, 0, 0, 0, 0, 0]])
    doc_lengths = torch.tensor([5, 4, 3, 5])

    adj, weights = GraphAdjacencyBuilder.build_adjacency(
        speakers, doc_lengths, seq_len, window_size=2
    )
    print(f"邻接矩阵形状: {adj.shape}")
    print(f"边权重形状: {weights.shape}")

    # 测试GATv2
    print("\n测试GATv2层...")
    gat = GATv2Layer(hidden_dim, hidden_dim, num_heads=8)
    node_features = torch.randn(batch_size, seq_len, hidden_dim)
    mask = MaskGenerator.create_padding_mask(lengths, seq_len)

    gat_output = gat(node_features, adj, weights, mask)
    print(f"GATv2输出形状: {gat_output.shape}")

    print("\n所有组件测试完成！")
