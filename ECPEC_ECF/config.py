# -*- coding: utf-8 -*-

import argparse
import os


class Config:
    """配置类"""

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='ECPEC')
        self._add_arguments()

    def _add_arguments(self):
        # 数据相关
        self.parser.add_argument('--dataset', type=str, default='meld',
                                choices=['iemocap', 'meld', 'dailydialog'], help='数据集名称')
        self.parser.add_argument('--data_dir', type=str, default='./data',
                                help='数据目录')

        # 模型架构
        self.parser.add_argument('--model_type', type=str, default='bert',
                                choices=['bilstm', 'bert'], help='模型类型')
        self.parser.add_argument('--hidden_dim', type=int, default=1024,
                                help='隐藏层维度')
        self.parser.add_argument('--n_heads', type=int, default=8,
                                help='多头注意力头数')
        self.parser.add_argument('--n_layers', type=int, default=2,
                                help='Transformer层数')

        # 数据维度
        self.parser.add_argument('--vocab_size', type=int, default=30000,
                                help='词汇表大小')
        self.parser.add_argument('--embedding_dim', type=int, default=300,
                                help='词嵌入维度')
        self.parser.add_argument('--position_embedding_dim', type=int, default=50,
                                help='位置嵌入维度')
        self.parser.add_argument('--max_doc_len', type=int, default=50,
                                help='最大对话长度')
        self.parser.add_argument('--max_sen_len', type=int, default=50,
                                help='最大句子长度')


        self.parser.add_argument('--use_emotion_category', action='store_true', default=False,help='是否使用情感类别')
        self.parser.add_argument('--use_emocate', action='store_true', default=False,help='是否使用细粒度情感分类（多类），否则使用二分类（neutral vs non-neutral）')

        # 训练相关
        self.parser.add_argument('--batch_size', type=int, default=64,
                                help='批次大小')
        self.parser.add_argument('--learning_rate', type=float, default=2e-5,
                                help='学习率')
        self.parser.add_argument('--weight_decay', type=float, default=5e-5,
                                help='权重衰减')
        self.parser.add_argument('--dropout', type=float, default=0.4,
                                help='Dropout率')
        self.parser.add_argument('--epochs', type=int, default=20,
                                help='训练轮数')
        self.parser.add_argument('--warmup_steps', type=int, default=20,
                                help='学习率预热步数')


        # 损失权重
        self.parser.add_argument('--emotion_weight', type=float, default=0.1,
                                help='情感检测损失权重')
        self.parser.add_argument('--cause_weight', type=float, default=0.2,
                                help='原因检测损失权重')

        self.parser.add_argument('--fgw_only', action='store_true', default=False,
                                help='启用后移除对级MLP，按每个情感中心在局部窗口内进行FGW对齐并以行级分布监督')
        self.parser.add_argument('--fgw_local_emo_window', type=int, default=2,
                                help='局部FGW时情感侧窗口半径W_e（窗口大小=2*W_e+1）')

        # FGW 预测解码策略
        self.parser.add_argument('--fgw_pred_strategy', type=str, default='topp',
                                 choices=['argmax', 'threshold', 'topk', 'topp'],
                                 help='FGW-only 解码策略：argmax/threshold/topk/topp')
        self.parser.add_argument('--fgw_threshold', type=float, default=0.5,
                                 help='当策略为threshold时使用的概率阈值')
        self.parser.add_argument('--fgw_top_k', type=int, default=2,
                                 help='当策略为topk时每行选择的列数k')
        self.parser.add_argument('--fgw_top_p', type=float, default=0.5,
                                 help='当策略为topp时的累计质量阈值p（0~1）')

        # 训练策略
        self.parser.add_argument('--share_encoder', action='store_true', default=True,
                                help='情感和原因是否共享编码器')
        self.parser.add_argument('--gradient_clip', type=float, default=1.0,
                                help='梯度裁剪阈值')
        self.parser.add_argument('--patience', type=int, default=8,
                                help='早停耐心')

        # 图神经网络相关
        self.parser.add_argument('--use_graph_encoder', action='store_true', default=True,
                                help='是否使用图神经网络编码器')
        self.parser.add_argument('--graph_num_layers', type=int, default=3,
                                help='图神经网络层数')
        self.parser.add_argument('--graph_window_size', type=int, default=5,
                                help='图神经网络时间窗口大小')
        self.parser.add_argument('--use_speaker_edges', action='store_true', default=True,
                                help='是否使用说话人边')
        self.parser.add_argument('--use_temporal_edges', action='store_true', default=True,
                                help='是否使用时间边')
        self.parser.add_argument('--distance_decay', action='store_true', default=True,
                                help='是否使用距离衰减权重')
        self.parser.add_argument('--graph_tau', type=float, default=2.0,
                                help='图神经网络距离衰减参数')

        # 设备和并行
        self.parser.add_argument('--device', type=str, default='cuda',
                                help='设备')
        self.parser.add_argument('--num_workers', type=int, default=4,
                                help='数据加载器工作线程数')

        # 预计算特征开关
        self.parser.add_argument('--feature_dir', type=str, default='./features',
                                help='离线特征缓存目录（内含各数据集子目录）')

        # 保存和加载
        self.parser.add_argument('--save_dir', type=str, default='./checkpoints',
                                help='模型保存目录')
        self.parser.add_argument('--log_dir', type=str, default='./logs',
                                help='日志目录')
        self.parser.add_argument('--load_checkpoint', type=str, default=None,
                                help='加载检查点路径')

        self.parser.add_argument('--weight_ratio_cap', type=float, default=5.0,
                                help='类别权重比例上限（正类/负类），避免过度偏向正类')

        # 其他
        self.parser.add_argument('--seed', type=int, default=42,
                                help='随机种子')
        self.parser.add_argument('--log_interval', type=int, default=10,
                                help='日志打印间隔')
        self.parser.add_argument('--eval_interval', type=int, default=1,
                                help='评估间隔（轮数）')
        self.parser.add_argument('--save_interval', type=int, default=5,
                                help='保存间隔（轮数）')
        self.parser.add_argument('--negative_sampling_ratio', type=float, default=4.0,
                                help='每个正对采样的负对个数（1:K）')

        # 候选方向控制：是否允许未来原因（False 表示仅保留过去/同句方向）
        self.parser.add_argument('--pred_future_cause', action='store_true', default=False,
                                help='是否允许原因出现在将来（默认False，仅过去/同句）')

        # 训练阶段：
        self.parser.add_argument('--train_max_pair_distance', type=int, default=5,
                                help='训练阶段保留 |cause-emotion| <= D 的候选对')
        # 评估阶段：确定性距离裁剪
        self.parser.add_argument('--eval_max_pair_distance', type=int, default=3,
                                help='验证/测试阶段仅保留 |cause-emotion| <= D 的候选对')

        self.parser.add_argument('--fgw_alpha', type=float, default=0.6,
                                help='FGW属性/结构折中系数')
        self.parser.add_argument('--fgw_window', type=int, default=4,
                                help='图内时间窗口大小')
        self.parser.add_argument('--fgw_tau', type=float, default=2.0,
                                help='图内距离衰减参数')
        self.parser.add_argument('--fgw_mode', type=str, default='blend', choices=['blend','replace'],
                                help='blend: 与pair prob加权融合；replace: 直接用FGW分数')
        self.parser.add_argument('--fgw_blend_lambda', type=float, default=0.4,
                                help='融合权重 λ，final = λ·FGW + (1-λ)·pair_prob')

        self.parser.add_argument('--fgw_blend_space', type=str, default='logit', choices=['prob', 'logit'],
                                help='融合空间：prob=线性概率融合；logit=对数几率融合(sigmoid(α·logit_fgw + (1-α)·logit_mlp))')


        # 评估/训练指标：FGW+MLP 融合后按行解码
        self.parser.add_argument('--fgw_fuse_row_decode', action='store_true', default=True,
                                help='评估/训练时启用：FGW与MLP线性融合后，按每情感行解码（argmax/threshold/topk/topp）')
        self.parser.add_argument('--mlp_temp', type=float, default=0.5,
                                help='MLP对分类logits的温度缩放T，用于概率标定（softmax(logits/T)）')

        # 训练期 OT 头蒸馏/融合
        self.parser.add_argument('--use_ot_head', action='store_true', default=True,
                                help='训练期启用OT头蒸馏/融合损失')
        self.parser.add_argument('--ot_lambda', type=float, default=0.4,
                                help='训练期损失权重 λ，L = λ·L_OT + (1-λ)·L_pair')
        self.parser.add_argument('--ot_loss', type=str, default='kl', choices=['bce','mse','kl'],
                                help='OT蒸馏损失类型：bce/mse/kl')
        self.parser.add_argument('--fgw_eps', type=float, default=0.4,
                                help='可微FGW头的熵正则系数')
        self.parser.add_argument('--fgw_iterations', type=int, default=10,
                                help='可微FGW头的外层迭代次数')
        self.parser.add_argument('--fgw_sinkhorn_iter', type=int, default=30,
                                help='可微FGW头 Sinkhorn 迭代次数')
        self.parser.add_argument('--fgw_sinkhorn_eps', type=float, default=1e-6,
                                help='Sinkhorn 稳定常数')
        self.parser.add_argument('--ot_sup_weight', type=float, default=0.3,
                                help='OT监督损失权重（可与蒸馏并用）')
        self.parser.add_argument('--fgw_row_norm', type=str, default='row_softmax', choices=['none','row_softmax','max'],
                                help='OT分数归一化方式：none/max/row_softmax')
        self.parser.add_argument('--fgw_row_temp', type=float, default=1.0,
                                help='row_softmax 的温度参数')

    def parse_args(self):
        args = self.parser.parse_args()

        # 数据集特定配置
        if args.dataset == 'iemocap':
            if args.use_emocate:
                args.n_emotions = 6
            else:
                args.n_emotions = 2
        elif args.dataset == 'meld':
            if args.use_emocate:
                args.n_emotions = 7
            else:
                args.n_emotions = 2
        elif args.dataset == 'dailydialog':
            if args.use_emocate:
                args.n_emotions = 7
            else:
                args.n_emotions = 2


        # 创建目录
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)

        return args


def get_model_config(args):
    """获取模型配置（纯文本）"""
    from models import ModelConfig
    config = ModelConfig(
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        position_embedding_dim=args.position_embedding_dim,
        max_sen_len=args.max_sen_len,
        max_doc_len=args.max_doc_len,
        n_heads=args.n_heads,
        graph_num_layers=args.graph_num_layers,
        graph_window_size=args.graph_window_size,
        use_speaker_edges=args.use_speaker_edges,
        use_temporal_edges=args.use_temporal_edges,
        distance_decay=args.distance_decay,
        graph_tau=args.graph_tau,
        use_emotion_category=args.use_emotion_category,

        n_emotions=args.n_emotions,
        use_ot_head=args.use_ot_head,
        fgw_alpha=args.fgw_alpha,
        fgw_eps=args.fgw_eps,
        fgw_iterations=args.fgw_iterations,
        fgw_sinkhorn_iter=args.fgw_sinkhorn_iter,
        fgw_sinkhorn_eps=args.fgw_sinkhorn_eps,

        pred_future_cause=args.pred_future_cause,
        train_max_pair_distance=args.train_max_pair_distance,
        fgw_only=getattr(args, 'fgw_only', False)
    )
    return config


if __name__ == "__main__":
    # 测试配置
    config = Config()
    args = config.parse_args()

    print("配置参数:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    print(f"\n数据集: {args.dataset}")
    print(f"情感类别数: {args.n_emotions}")
