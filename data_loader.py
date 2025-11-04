# -*- coding: utf-8 -*-

import os

import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from collections import defaultdict

class EncodedFeature_Dataset(Dataset):

    def __init__(self, json_file_path, feature_file_path, dataset_type='iemocap',
                 max_doc_len=35, pred_future_cause=False, use_emocate=False,
                 max_pair_distance=None):
        import torch as _torch
        self.json_file_path = json_file_path
        self.feature_file_path = feature_file_path
        self.dataset_type = dataset_type
        self.max_doc_len = max_doc_len
        self.pred_future_cause = pred_future_cause
        self.use_emocate = use_emocate
        self.max_pair_distance = max_pair_distance

        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)

        bundle = _torch.load(self.feature_file_path, map_location='cpu')
        self.feature_map = bundle['features']  # conv_id -> {features, doc_len}
        self.hidden_dim = int(bundle['meta']['hidden_dim'])
        self.true_pairs_by_conv_full = {}

        # Emotion mapping
        if dataset_type == 'iemocap':
            self.emotion_mapping = {
                'neutral': 0, 'anger': 1, 'fear': 2,
                'happiness': 3, 'sadness': 4, 'frustration': 5
            }
        elif dataset_type == 'meld':
            self.emotion_mapping = {
                'neutral': 0, 'anger': 1, 'disgust': 2, 'fear': 3,
                'joy': 4, 'sadness': 5, 'surprise': 6
            }
        else:  # dailydialog
            self.emotion_mapping = {
                'neutral': 0, 'happiness': 1, 'surprise': 2,
                'anger': 3, 'sadness': 4, 'disgust': 5, 'fear': 6
            }

        self.data = []
        for conv in self.raw_data:
            conv_id = str(conv.get('conversation_ID'))
            utterances = conv.get('conversation', [])
            texts = [utt.get('text', '') for utt in utterances]
            speakers = [utt.get('speaker', '') for utt in utterances]
            emotions_str = [utt.get('emotion', 'neutral') for utt in utterances]
            emotion_labels_full = [self.emotion_mapping.get(e, 0) for e in emotions_str]

            pairs_raw = conv.get('emotion-cause_pairs', [])
            emotion_cause_pairs = []
            for emo_str, cause_str in pairs_raw:
                try:
                    emo_id = int(emo_str.split('_', 1)[0])
                    cause_id = int(cause_str.split('_', 1)[0])
                    emotion_cause_pairs.append((emo_id, cause_id))
                except Exception:
                    continue

            base_item = {
                'conversation_id': conv_id,
                'texts': texts,
                'speakers': speakers,
                'emotion_labels': emotion_labels_full,
                'emotion_cause_pairs': emotion_cause_pairs
            }

            item = self._prepare_dialogue(base_item)
            self.data.append(item)

            doc_len = min(len(texts), self.max_doc_len)
            filtered_pairs = []
            for (emo_id, cause_id) in emotion_cause_pairs:
                if emo_id <= doc_len and cause_id <= doc_len:
                    if (self.pred_future_cause or cause_id <= emo_id):
                        if self.max_pair_distance is None or abs(cause_id - emo_id) <= int(self.max_pair_distance):
                            filtered_pairs.append((conv_id, emo_id, cause_id))
            self.true_pairs_by_conv_full[conv_id] = filtered_pairs

    def _prepare_dialogue(self, data_item):
        import torch as _torch
        texts = data_item['texts']
        emotion_cause_pairs = data_item['emotion_cause_pairs']
        speakers = data_item['speakers']
        conv_id = data_item['conversation_id']

        doc_len = min(len(texts), self.max_doc_len)

        pair_indices, pair_distances, pair_labels = [], [], []
        positive_set = set([
            (emo_id, cause_id)
            for (emo_id, cause_id) in emotion_cause_pairs
            if emo_id <= doc_len and cause_id <= doc_len
        ])
        for emo_id in range(1, doc_len + 1):
            for cause_id in range(1, doc_len + 1):
                if not self.pred_future_cause and emo_id < cause_id:
                    continue
                pair_indices.append((emo_id - 1, cause_id - 1))
                pair_distances.append(cause_id - emo_id)
                pair_labels.append(1 if (emo_id, cause_id) in positive_set else 0)

        pair_indices = _torch.tensor(pair_indices, dtype=_torch.long)
        pair_distances = _torch.tensor(pair_distances, dtype=_torch.long)
        pair_labels = _torch.tensor(pair_labels, dtype=_torch.long)

        if self.max_pair_distance is not None:
            dist_mask = pair_distances.abs() <= int(self.max_pair_distance)
            if dist_mask.any():
                pair_indices = pair_indices[dist_mask]
                pair_distances = pair_distances[dist_mask]
                pair_labels = pair_labels[dist_mask]

        # Utterance-level labels
        emotion_labels = _torch.zeros(self.max_doc_len, dtype=_torch.long)
        cause_labels = _torch.zeros(self.max_doc_len, dtype=_torch.long)
        for idx in range(doc_len):
            if self.use_emocate:
                emotion_labels[idx] = data_item['emotion_labels'][idx]
            else:
                emotion_labels[idx] = 1 if data_item['emotion_labels'][idx] > 0 else 0
        cause_utterances = set([c for (_, c) in emotion_cause_pairs if c <= doc_len])
        for idx in range(doc_len):
            cause_labels[idx] = 1 if (idx + 1) in cause_utterances else 0

        # Speaker encoding
        unique_speakers = list(set(speakers[:doc_len]))
        speaker_to_id = {spk: idx for idx, spk in enumerate(unique_speakers)}
        speaker_ids = _torch.zeros(self.max_doc_len, dtype=_torch.long)
        for i in range(doc_len):
            speaker_ids[i] = speaker_to_id[speakers[i]]

        # Precomputed features
        feat_entry = self.feature_map.get(conv_id)
        if feat_entry is None:
            raise KeyError(f'Missing features for conversation_id={conv_id}')
        precomputed = feat_entry['features']  # (L, D)
        if precomputed.size(0) != self.max_doc_len:
            # Align to max_doc_len
            L, D = precomputed.size(0), precomputed.size(1)
            mat = _torch.zeros(self.max_doc_len, D, dtype=precomputed.dtype)
            mat[:min(L, self.max_doc_len)] = precomputed[:min(L, self.max_doc_len)]
            precomputed = mat

        return {
            'precomputed_features': precomputed,  # (L, D)
            'doc_len': doc_len,
            'speakers': speaker_ids,
            'emotion_labels': emotion_labels,
            'cause_labels': cause_labels,
            'pair_indices': pair_indices,
            'pair_distances': pair_distances,
            'pair_labels': pair_labels,
            'texts': texts,  # For potential doc encoding, not actually read
            'conversation_id': conv_id
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn_features(batch):
    import torch as _torch
    input_feats = _torch.stack([item['precomputed_features'] for item in batch])  
    doc_len = _torch.tensor([item['doc_len'] for item in batch], dtype=_torch.long)
    speakers = _torch.stack([item['speakers'] for item in batch])
    emotion_labels = _torch.stack([item['emotion_labels'] for item in batch])
    cause_labels = _torch.stack([item['cause_labels'] for item in batch])

    pair_indices = []
    pair_distances = []
    pair_labels = []
    conversation_ids = []
    for b_idx, item in enumerate(batch):
        idxs = item['pair_indices']
        num_pairs = idxs.size(0)
        bcol = _torch.full((num_pairs, 1), b_idx, dtype=_torch.long)
        pair_indices.append(_torch.cat([bcol, idxs], dim=1))
        pair_distances.append(item['pair_distances'])
        pair_labels.append(item['pair_labels'])
        conversation_ids.extend([item['conversation_id']] * num_pairs)

    pair_indices = _torch.cat(pair_indices, dim=0)
    pair_distances = _torch.cat(pair_distances, dim=0)
    pair_labels = _torch.cat(pair_labels, dim=0)

    return {
        'precomputed_features': input_feats,
        'doc_len': doc_len,
        'speakers': speakers,
        'emotion_labels': emotion_labels,
        'cause_labels': cause_labels,
        'pair_indices': pair_indices,
        'pair_distances': pair_distances,
        'pair_labels': pair_labels,
        'texts': [item['texts'] for item in batch],
        'conversation_id': [item['conversation_id'] for item in batch],
        'pair_conversation_id': conversation_ids
    }


def create_feature_data_loaders(dataset_name='iemocap', feature_dir='./features', batch_size=8,pred_future_cause=False, use_emocate=False,eval_max_pair_distance=None, train_max_pair_distance=None,max_doc_len=50):

    if dataset_name == 'iemocap':
        train_json = 'data/iemocap/train.json'
        test_json = 'data/iemocap/test.json'
        dev_json = None
    elif dataset_name == 'meld':
        train_json = 'data/meld/train.json'
        test_json = 'data/meld/test.json'
        dev_json = 'data/meld/dev.json'
    else:
        train_json = 'data/dailydialog/train.json'
        test_json = 'data/dailydialog/test.json'
        dev_json = 'data/dailydialog/dev.json'

    base_dir = os.path.join(feature_dir, dataset_name)
    train_feat = os.path.join(base_dir, 'train_features.pt')
    test_feat = os.path.join(base_dir, 'test_features.pt')
    dev_feat = os.path.join(base_dir, 'dev_features.pt') if dev_json else None


    train_dataset = EncodedFeature_Dataset(
        train_json, train_feat, dataset_name,
        max_doc_len=max_doc_len,
        pred_future_cause=pred_future_cause, use_emocate=use_emocate,
        max_pair_distance=train_max_pair_distance
    )

    test_dataset = EncodedFeature_Dataset(
        test_json, test_feat, dataset_name,
        max_doc_len=max_doc_len,
        pred_future_cause=pred_future_cause, use_emocate=use_emocate,
        max_pair_distance=eval_max_pair_distance
    )

    vocab_size = 0 

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_features)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_features)

    dev_loader = None
    if dev_json:
        dev_dataset = EncodedFeature_Dataset(
            dev_json, dev_feat, dataset_name,
            max_doc_len=max_doc_len,
            pred_future_cause=pred_future_cause, use_emocate=use_emocate,
            max_pair_distance=eval_max_pair_distance
        )
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_features)

    return train_loader, test_loader, dev_loader, vocab_size
