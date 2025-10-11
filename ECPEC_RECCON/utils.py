# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import time
import os
import random
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from collections import defaultdict


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_time():
    """打印当前时间"""
    print(f'\n----------{time.strftime("%Y-%m-%d %X", time.localtime())}----------')


def list_round(a_list, decimals=4):
    """列表数值四舍五入"""
    return [round(float(i), decimals) for i in a_list]


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


class MetricsCalculator:
    """评估指标计算器"""
    
    @staticmethod
    def calculate_prf(pred, true, mask=None, average='binary', use_emocate=False):
        """
        计算精确率、召回率、F1分数

        Args:
            pred: 预测结果 (batch_size, seq_len) 或 (batch_size,)
            true: 真实标签 (batch_size, seq_len) 或 (batch_size,)
            mask: 掩码 (batch_size, seq_len) 或 None
            average: 'binary', 'macro', 'micro', 'weighted'
            use_emocate: 是否使用细粒度情感分类，影响评价方式选择
        """
        if len(pred.shape) > 1:
            # 序列标注任务
            pred_flat = []
            true_flat = []
            
            for i in range(pred.shape[0]):
                if mask is not None:
                    length = mask[i].sum().item()
                    pred_flat.extend(pred[i][:length].tolist())
                    true_flat.extend(true[i][:length].tolist())
                else:
                    pred_flat.extend(pred[i].tolist())
                    true_flat.extend(true[i].tolist())
        else:
            # 分类任务
            pred_flat = pred.tolist()
            true_flat = true.tolist()
        
        # 自动检测任务类型
        unique_labels = set(true_flat)
        n_classes = len(unique_labels)

        # 根据use_emocate和实际类别数选择合适的评价方式
        if use_emocate:
            # 细粒度情感分类：使用macro平均（排除neutral类）或weighted平均
            if average == 'binary':
                average = 'macro'  # 多类情况下改为macro
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_flat, pred_flat, average=average, zero_division=0
            )
        else:
            # 二分类模式：neutral vs non-neutral
            if n_classes == 2 and average == 'binary':
                precision, recall, f1, _ = precision_recall_fscore_support(
                    true_flat, pred_flat, average='binary', zero_division=0
                )
            elif n_classes > 2 and average == 'binary':
                # 多分类但指定了binary，改为macro（向后兼容）
                precision, recall, f1, _ = precision_recall_fscore_support(
                    true_flat, pred_flat, average='macro', zero_division=0
                )
            else:
                # 使用指定的average方式
                precision, recall, f1, _ = precision_recall_fscore_support(
                    true_flat, pred_flat, average=average, zero_division=0
                )
        
        return precision, recall, f1
    
    @staticmethod
    def calculate_emotion_category_f1(pred, true, mask=None, emotion_names=None):
        """
        计算多类情感分类的F1分数
        
        Args:
            pred: 预测结果 (batch_size, seq_len)
            true: 真实标签 (batch_size, seq_len)
            mask: 掩码 (batch_size, seq_len)
            emotion_names: 情感类别名称列表
        """
        pred_flat = []
        true_flat = []
        
        for i in range(pred.shape[0]):
            if mask is not None:
                length = mask[i].sum().item()
                pred_flat.extend(pred[i][:length].tolist())
                true_flat.extend(true[i][:length].tolist())
            else:
                pred_flat.extend(pred[i].tolist())
                true_flat.extend(true[i].tolist())
        
        # 计算混淆矩阵
        cm = confusion_matrix(true_flat, pred_flat)
        
        # 计算每个类别的P, R, F1
        n_classes = cm.shape[0]
        precision = np.zeros(n_classes)
        recall = np.zeros(n_classes)
        f1 = np.zeros(n_classes)
        
        for i in range(n_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            precision[i] = tp / (tp + fp + 1e-8)
            recall[i] = tp / (tp + fn + 1e-8)
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-8)
        
        # 计算加权平均（排除中性情感类别0）
        if n_classes > 1:
            weights = cm[1:, :].sum(axis=1)  # 排除neutral类别
            total_weight = weights.sum()
            if total_weight > 0:
                weighted_f1 = np.sum(f1[1:] * weights) / total_weight
            else:
                weighted_f1 = 0.0
        else:
            weighted_f1 = f1[0]
        
        results = {
            'per_class_f1': f1,
            'weighted_f1': weighted_f1,
            'macro_f1': f1[1:].mean() if n_classes > 1 else f1[0],  # 排除neutral
            'confusion_matrix': cm
        }
        
        if emotion_names:
            results['emotion_names'] = emotion_names
            for i, name in enumerate(emotion_names):
                results[f'{name}_f1'] = f1[i]
        
        return results


class PairEvaluator:
    """端到端对提取评估器"""
    
    @staticmethod
    def evaluate_pairs(all_true_pairs, all_pred_pairs):
        """
        评估情感-原因对提取结果
        
        Args:
            all_true_pairs: 真实对列表 [[(conv_id, emo_id, cause_id), ...], ...]
            all_pred_pairs: 预测对列表 [[(conv_id, emo_id, cause_id), ...], ...]
        """
        # 展平所有对
        true_pairs_flat = []
        pred_pairs_flat = []
        
        for true_pairs, pred_pairs in zip(all_true_pairs, all_pred_pairs):
            true_pairs_flat.extend(true_pairs)
            pred_pairs_flat.extend(pred_pairs)
        
        # 转换为集合
        true_set = set(true_pairs_flat)
        pred_set = set(pred_pairs_flat)
        
        # 计算指标
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'n_true_pairs': len(true_pairs_flat),
            'n_pred_pairs': len(pred_pairs_flat)
        }
    
    @staticmethod
    def extract_pairs_from_predictions(conversation_ids, emo_ids, cause_ids, 
                                     predictions, threshold=0.5):
        """
        
        Args:
            conversation_ids: 对话ID列表
            emo_ids: 情感话语ID列表
            cause_ids: 原因话语ID列表
            predictions: 预测概率 (N, 2) 或 预测标签 (N,)
            threshold: 阈值
        """
        if predictions.dim() == 2:
            # 概率预测
            pred_labels = (predictions[:, 1] > threshold).int()
        else:
            # 已经是标签
            pred_labels = predictions
        
        pairs = []
        for i, pred in enumerate(pred_labels):
            if pred == 1:
                pairs.append((conversation_ids[i], emo_ids[i].item(), cause_ids[i].item()))
        
        return pairs


class LossWithRegularization(nn.Module):
    """带正则化的损失函数"""
    
    def __init__(self, base_criterion, l2_lambda=1e-5):
        super(LossWithRegularization, self).__init__()
        self.base_criterion = base_criterion
        self.l2_lambda = l2_lambda
    
    def forward(self, outputs, targets, model):
        # 基础损失
        base_loss = self.base_criterion(outputs, targets)
        
        # L2正则化
        l2_reg = 0
        for param in model.parameters():
            l2_reg += torch.norm(param, 2)
        
        total_loss = base_loss + self.l2_lambda * l2_reg
        
        return total_loss, base_loss, l2_reg


class Step1Loss(nn.Module):
    """Step1的多任务损失函数"""
    
    def __init__(self, emotion_weight=1.0, cause_weight=1.0, 
                 audio_emotion_weight=0.5, visual_emotion_weight=0.5):
        super(Step1Loss, self).__init__()
        self.emotion_weight = emotion_weight
        self.cause_weight = cause_weight
        self.audio_emotion_weight = audio_emotion_weight
        self.visual_emotion_weight = visual_emotion_weight
        
        # 情感检测可能需要处理类别不平衡
        self.emotion_criterion = nn.CrossEntropyLoss()
        self.cause_criterion = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets, mask=None):
        """
        Args:
            outputs: 模型输出字典
            targets: 目标字典 {'y_emotion': ..., 'y_cause': ...}
            mask: 序列掩码
        """
        emotion_logits = outputs['emotion_logits']  # (batch_size, seq_len, n_emotions)
        cause_logits = outputs['cause_logits']      # (batch_size, seq_len, 2)
        
        y_emotion = targets['y_emotion']  # (batch_size, seq_len)
        y_cause = targets['y_cause']      # (batch_size, seq_len)
        
        # 展平用于计算损失
        if mask is not None:
            # 只计算有效位置的损失
            emotion_logits_flat = emotion_logits[mask]
            cause_logits_flat = cause_logits[mask]
            y_emotion_flat = y_emotion[mask]
            y_cause_flat = y_cause[mask]
        else:
            emotion_logits_flat = emotion_logits.view(-1, emotion_logits.size(-1))
            cause_logits_flat = cause_logits.view(-1, cause_logits.size(-1))
            y_emotion_flat = y_emotion.view(-1)
            y_cause_flat = y_cause.view(-1)
        
        # 主要损失
        emotion_loss = self.emotion_criterion(emotion_logits_flat, y_emotion_flat.long())
        cause_loss = self.cause_criterion(cause_logits_flat, y_cause_flat.long())
        
        total_loss = (self.emotion_weight * emotion_loss + 
                     self.cause_weight * cause_loss)
        
        loss_dict = {
            'total_loss': total_loss,
            'emotion_loss': emotion_loss,
            'cause_loss': cause_loss
        }
        
        # 辅助模态损失
        if 'audio_emotion_logits' in outputs and outputs['audio_emotion_logits'] is not None:
            audio_emotion_logits = outputs['audio_emotion_logits']
            if mask is not None:
                audio_emotion_logits_flat = audio_emotion_logits[mask]
            else:
                audio_emotion_logits_flat = audio_emotion_logits.view(-1, audio_emotion_logits.size(-1))
            
            audio_emotion_loss = self.emotion_criterion(audio_emotion_logits_flat, y_emotion_flat.long())
            total_loss += self.audio_emotion_weight * audio_emotion_loss
            loss_dict['audio_emotion_loss'] = audio_emotion_loss
        
        if 'visual_emotion_logits' in outputs and outputs['visual_emotion_logits'] is not None:
            visual_emotion_logits = outputs['visual_emotion_logits']
            if mask is not None:
                visual_emotion_logits_flat = visual_emotion_logits[mask]
            else:
                visual_emotion_logits_flat = visual_emotion_logits.view(-1, visual_emotion_logits.size(-1))
            
            visual_emotion_loss = self.emotion_criterion(visual_emotion_logits_flat, y_emotion_flat.long())
            total_loss += self.visual_emotion_weight * visual_emotion_loss
            loss_dict['visual_emotion_loss'] = visual_emotion_loss
        
        loss_dict['total_loss'] = total_loss
        
        return loss_dict


class ModelSaver:
    """模型保存器"""
    
    def __init__(self, save_dir, max_checkpoints=5):
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        os.makedirs(save_dir, exist_ok=True)
        
    def save_checkpoint(self, model, optimizer, epoch, metrics, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        # 保存当前检查点
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
        
        # 清理旧检查点
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """清理旧的检查点"""
        checkpoint_files = []
        for file in os.listdir(self.save_dir):
            if file.startswith('checkpoint_epoch_') and file.endswith('.pt'):
                checkpoint_files.append(file)
        
        if len(checkpoint_files) > self.max_checkpoints:
            # 按文件名排序（包含epoch信息）
            checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            # 删除最旧的文件
            for file in checkpoint_files[:-self.max_checkpoints]:
                os.remove(os.path.join(self.save_dir, file))
    
    def load_checkpoint(self, model, optimizer=None, filename='best_model.pt'):
        """加载检查点"""
        checkpoint_path = os.path.join(self.save_dir, filename)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint['epoch'], checkpoint['metrics']
        else:
            print(f"检查点文件 {checkpoint_path} 不存在")
            return None, None


if __name__ == "__main__":
    # 测试工具函数
    print("测试评估指标计算...")
    
    # 测试二分类指标
    pred = torch.tensor([1, 0, 1, 1, 0])
    true = torch.tensor([1, 0, 0, 1, 0])
    
    p, r, f1 = MetricsCalculator.calculate_prf(pred, true)
    print(f"二分类 - P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}")
    
    # 测试序列标注指标
    pred_seq = torch.tensor([[1, 0, 1, 0], [0, 1, 1, 0]])
    true_seq = torch.tensor([[1, 0, 0, 0], [0, 1, 1, 0]])
    mask_seq = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    
    p, r, f1 = MetricsCalculator.calculate_prf(pred_seq, true_seq, mask_seq)
    print(f"序列标注 - P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}")
    
    # 测试情感分类指标
    emotion_pred = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]])
    emotion_true = torch.tensor([[0, 1, 1, 0], [1, 2, 0, 2]])
    emotion_names = ['neutral', 'anger', 'joy']
    
    results = MetricsCalculator.calculate_emotion_category_f1(
        emotion_pred, emotion_true, mask_seq, emotion_names
    )
    print(f"情感分类 - 加权F1: {results['weighted_f1']:.4f}")
    
    print("\n工具函数测试完成！")
