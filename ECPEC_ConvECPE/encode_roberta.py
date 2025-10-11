# -*- coding: utf-8 -*-

"""
RoBERTa 离线文本特征提取脚本

功能：
- 读取指定数据集的 JSON（train/dev/test）
- 使用 RoBERTa + tokenizer 对每个话语逐句编码
- 采用 mean pooling 得到句向量 (hidden_dim)
- 将每个对话的句向量堆叠为 (max_doc_len, hidden_dim) 并保存为 .pt
- 每个 split 生成一个 features 文件，结构：
    {
        'meta': {
            'model_name': str,
            'hidden_dim': int,
            'tokenizer_len': int,
            'max_doc_len': int,
            'max_sen_len': int,
            'pooling': 'mean'
        },
        'index': [conv_id1, conv_id2, ...],
        'features': {
            conv_id: {
                'features': Tensor (max_doc_len, hidden_dim),
                'doc_len': int
            },
            ...
        }
    }

用法示例：
python MECPE/offline_encode_roberta.py \
  --dataset meld \
  --data_dir MECPE/data \
  --bert_model_name ../roberta \
  --feature_dir MECPE/features \
  --max_doc_len 50 \
  --max_sen_len 50 \
  --batch_size 64
"""

import os
import json
import argparse
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from model_path_utils import ensure_model_path_exists


class UtteranceDataset(Dataset):
    """将 JSON 数据展开为逐句条目，便于批量前向。"""

    def __init__(self, json_path: str, max_doc_len: int, max_sen_len: int):
        self.json_path = json_path
        self.max_doc_len = max_doc_len
        self.max_sen_len = max_sen_len
        with open(json_path, 'r', encoding='utf-8') as f:
            self.raw = json.load(f)
        # 建立 (conv_idx, utt_idx, text) 列表，并记录每个 conv 的起止范围
        self.items = []  # (conv_id, utt_idx, text)
        self.conv_ranges = {}  # conv_id -> (start, end)
        cursor = 0
        for conv in self.raw:
            conv_id = str(conv.get('conversation_ID'))
            texts = [utt.get('text', '') for utt in conv.get('conversation', [])]
            doc_len = min(len(texts), self.max_doc_len)
            start = cursor
            for i in range(doc_len):
                self.items.append((conv_id, i, texts[i]))
                cursor += 1
            end = cursor
            self.conv_ranges[conv_id] = (start, end)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        conv_id, utt_idx, text = self.items[idx]
        return {
            'conv_id': conv_id,
            'utt_idx': utt_idx,
            'text': text
        }


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """对 token 维做 masked mean pooling。
    last_hidden_state: (B, T, H)
    attention_mask:    (B, T)
    return: (B, H)
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def build_split_paths(data_dir: str, dataset: str):
    if dataset == 'iemocap':
        train = os.path.join(data_dir, 'iemocap', 'train.json')
        test = os.path.join(data_dir, 'iemocap', 'test.json')
        dev = None
    elif dataset == 'meld':
        train = os.path.join(data_dir, 'meld', 'train.json')
        test = os.path.join(data_dir, 'meld', 'test.json')
        dev = os.path.join(data_dir, 'meld', 'dev.json')
    else:  # dailydialog
        train = os.path.join(data_dir, 'dailydialog', 'train.json')
        test = os.path.join(data_dir, 'dailydialog', 'test.json')
        # 纠正原仓库中 vaild.json 的拼写问题，这里使用 dev.json 作为约定
        dev = os.path.join(data_dir, 'dailydialog', 'dev.json')
    return train, dev, test


def encode_split(json_path: str,
                 model: AutoModel,
                 tokenizer: AutoTokenizer,
                 device: torch.device,
                 max_doc_len: int,
                 max_sen_len: int,
                 batch_size: int = 64) -> Dict[str, Any]:
    ds = UtteranceDataset(json_path, max_doc_len, max_sen_len)

    def collate(batch: List[Dict[str, Any]]):
        texts = [b['text'] for b in batch]
        enc = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_sen_len,
            return_tensors='pt'
        )
        return {
            'conv_id': [b['conv_id'] for b in batch],
            'utt_idx': torch.tensor([b['utt_idx'] for b in batch], dtype=torch.long),
            'input_ids': enc['input_ids'],
            'attention_mask': enc['attention_mask']
        }

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    hidden_dim = model.config.hidden_size
    # 暂存：conv_id -> list[(utt_idx, feat)]
    tmp: Dict[str, List[torch.Tensor]] = {}

    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state  # (B, T, H)
            sent_feat = mean_pool(last_hidden, attention_mask)  # (B, H)
            for i, conv_id in enumerate(batch['conv_id']):
                utt_idx = int(batch['utt_idx'][i].item())
                feat = sent_feat[i].cpu()
                tmp.setdefault(conv_id, []).append((utt_idx, feat))

    # 组装为 (max_doc_len, H)
    features: Dict[str, Dict[str, Any]] = {}
    for conv in ds.raw:
        conv_id = str(conv.get('conversation_ID'))
        texts = [utt.get('text', '') for utt in conv.get('conversation', [])]
        doc_len = min(len(texts), max_doc_len)
        mat = torch.zeros(max_doc_len, hidden_dim, dtype=torch.float16)
        if conv_id in tmp:
            pairs = sorted(tmp[conv_id], key=lambda x: x[0])
            for (idx, feat) in pairs:
                if idx < max_doc_len:
                    mat[idx] = feat.to(dtype=torch.float16)
        features[conv_id] = {
            'features': mat,
            'doc_len': int(doc_len)
        }

    index = list(features.keys())
    return {
        'meta': {
            'model_name': tokenizer.name_or_path,
            'hidden_dim': hidden_dim,
            'tokenizer_len': len(tokenizer),
            'max_doc_len': max_doc_len,
            'max_sen_len': max_sen_len,
            'pooling': 'mean'
        },
        'index': index,
        'features': features
    }


def main():
    parser = argparse.ArgumentParser(description='RoBERTa Offline Utterance Encoder (mean pooling)')
    parser.add_argument('--dataset', type=str, default='meld', choices=['iemocap', 'meld', 'dailydialog'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--bert_model_name', type=str, default='../roberta')
    parser.add_argument('--feature_dir', type=str, default='./features')
    parser.add_argument('--max_doc_len', type=int, default=50)
    parser.add_argument('--max_sen_len', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    model_name = ensure_model_path_exists(args.bert_model_name)
    os.makedirs(args.feature_dir, exist_ok=True)
    out_dir = os.path.join(args.feature_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_json, dev_json, test_json = build_split_paths(args.data_dir, args.dataset)

    def run_and_save(split_name: str, json_path: str):
        if json_path is None:
            return
        print(f'Encoding split: {split_name} from {json_path}')
        bundle = encode_split(
            json_path=json_path,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_doc_len=args.max_doc_len,
            max_sen_len=args.max_sen_len,
            batch_size=args.batch_size
        )
        out_path = os.path.join(out_dir, f'{split_name}_features.pt')
        torch.save(bundle, out_path)
        print(f'Saved: {out_path} (index={len(bundle["index"])})')

    run_and_save('train', train_json)
    run_and_save('dev', dev_json)
    run_and_save('test', test_json)

    print('Done.')


if __name__ == '__main__':
    main()

