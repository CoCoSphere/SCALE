# -*- coding: utf-8 -*-
import argparse
import os


class Config:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='ECPEC')
        self._add_arguments()

    def _add_arguments(self):
        # Data and paths
        self.parser.add_argument('--dataset', type=str, default='meld', choices=['iemocap', 'meld', 'dailydialog'])
        self.parser.add_argument('--data_dir', type=str, default='./data')
        self.parser.add_argument('--feature_dir', type=str, default='./features')
        self.parser.add_argument('--save_dir', type=str, default='./checkpoints')
        self.parser.add_argument('--log_dir', type=str, default='./logs')

        # Device & training control
        self.parser.add_argument('--device', type=str, default='cuda')
        self.parser.add_argument('--seed', type=int, default=42)
        self.parser.add_argument('--log_interval', type=int, default=10)
        self.parser.add_argument('--eval_interval', type=int, default=1)
        self.parser.add_argument('--save_interval', type=int, default=1)
        self.parser.add_argument('--patience', type=int, default=8)
        self.parser.add_argument('--load_checkpoint', type=str, default='')

        # Training hyperparameters
        self.parser.add_argument('--batch_size', type=int, default=128)
        self.parser.add_argument('--learning_rate', type=float, default=4e-5)
        self.parser.add_argument('--weight_decay', type=float, default=4e-5)
        self.parser.add_argument('--epochs', type=int, default=20)
        self.parser.add_argument('--dropout', type=float, default=0.3)

        # Model architecture
        self.parser.add_argument('--hidden_dim', type=int, default=1024)
        self.parser.add_argument('--n_heads', type=int, default=8)
        self.parser.add_argument('--max_doc_len', type=int, default=50)
        self.parser.add_argument('--position_embedding_dim', type=int, default=50)
        
        # Node-level embeddings
        self.parser.add_argument('--use_pos_embed', action='store_true', default=True,help='Use absolute position embeddings')
        self.parser.add_argument('--use_speaker_embed', action='store_true', default=True,help='Use speaker embeddings')
        self.parser.add_argument('--speaker_vocab_size', type=int, default=16,
                        help='Maximum number of speakers in a conversation')

        # Graph structure (window/speaker/temporal)
        self.parser.add_argument('--graph_num_layers', type=int, default=2)
        self.parser.add_argument('--graph_window_size', type=int, default=5)
        self.parser.add_argument('--use_speaker_edges', action='store_true', default=True)
        self.parser.add_argument('--use_temporal_edges', action='store_true',default=True)
        self.parser.add_argument('--distance_decay', action='store_true', default=True)
        self.parser.add_argument('--graph_tau', type=float, default=2.0)

       
        # ReduceLROnPlateau hyperparameters (replacing linear warmup scheduler)
        self.parser.add_argument('--plateau_factor', type=float, default=0.5)
        self.parser.add_argument('--plateau_patience', type=int, default=2)
        self.parser.add_argument('--plateau_threshold', type=float, default=1e-3)
        self.parser.add_argument('--plateau_cooldown', type=int, default=1)
        self.parser.add_argument('--plateau_min_lr', type=float, default=1e-6)
        self.parser.add_argument('--gradient_clip', type=float, default=1.0)


        # Global decoding (MCMF) parameters
        self.parser.add_argument('--mcmf_row_capacity', type=int, default=2)
        self.parser.add_argument('--mcmf_col_capacity', type=int, default=3)
        self.parser.add_argument('--mcmf_lambda', type=float, default=0.5, help='PC-bmatching threshold price in prob/logit space')
        self.parser.add_argument('--mcmf_score_space', type=str, default='prob', choices=['prob', 'logit'])
        self.parser.add_argument('--mcmf_eps', type=float, default=1e-6)

        # MCMF edge pre-filtering: keep top-K per row with score >= threshold
        self.parser.add_argument('--mcmf_pre_topk_per_row', type=int, default=3)
        self.parser.add_argument('--mcmf_pre_min_prob', type=float, default=0.1)
        self.parser.add_argument('--mcmf_pre_min_logit', type=float, default=None)

        # Task and labels
        self.parser.add_argument('--use_emocate', action='store_true', default=False)
        self.parser.add_argument('--pred_future_cause', action='store_true', default=False)
        self.parser.add_argument('--train_max_pair_distance', type=int, default=5)
        self.parser.add_argument('--eval_max_pair_distance', type=int, default=4)
        self.parser.add_argument('--weight_ratio_cap', type=float, default=5.0)

        # Loss weights
        self.parser.add_argument('--emotion_weight', type=float, default=0.1)
        self.parser.add_argument('--cause_weight', type=float, default=0.3)

        # FGW / OT head
        self.parser.add_argument('--use_ot_head', action='store_true', default=True)
        self.parser.add_argument('--ot_lambda', type=float, default=0.4)
        self.parser.add_argument('--ot_loss', type=str, default='kl', choices=['bce', 'mse', 'kl'])

        self.parser.add_argument('--fgw_eps', type=float, default=0.1)
        self.parser.add_argument('--fgw_iterations', type=int, default=12)
        self.parser.add_argument('--fgw_sinkhorn_iter', type=int, default=50)
        self.parser.add_argument('--fgw_sinkhorn_eps', type=float, default=1e-6)
        self.parser.add_argument('--fgw_row_norm', type=str, default='entmax', 
                                 choices=['none', 'row_softmax', 'entmax', 'sparsemax', 'max'],
                                 help='Row normalization method')
        self.parser.add_argument('--fgw_row_temp', type=float, default=1.0,
                                 help='Temperature coefficient before normalization (for softmax/entmax/sparsemax)')
        self.parser.add_argument('--fgw_entmax_alpha', type=float, default=1.7,
                                 help='Entmax alpha parameter')
        self.parser.add_argument('--fgw_entmax_bisect_iter', type=int, default=50,
                                 help='Entmax bisection search iterations')

        
        # UOT (unbalanced) and capacity/consistency related
        self.parser.add_argument('--fgw_use_uot', action='store_true', default=True)
        self.parser.add_argument('--fgw_uot_rho', type=float, default=2.0)

        # FGW fusion
        self.parser.add_argument('--mlp_temp', type=float, default=2.5)
        self.parser.add_argument('--fgw_alpha', type=float, default=0.4)
        self.parser.add_argument('--fgw_mode', type=str, default='blend', choices=['blend','replace'])
        self.parser.add_argument('--fgw_blend_lambda', type=float, default=0.2)
        self.parser.add_argument('--fgw_blend_space', type=str, default='logit', choices=['prob','logit'])
        # Semantic cost: 1-cos or Mahalanobis (diagonal)
        self.parser.add_argument('--fgw_attr_metric', type=str, default='maha', choices=['cos','maha'])
        
        # Additional training loss weights: transport supervision, distillation with T
        self.parser.add_argument('--transport_ce_lambda', type=float, default=0.4)
        self.parser.add_argument('--transport_ce_pos_weight', type=float, default=2.0)
        self.parser.add_argument('--distill_with_t_lambda', type=float, default=0.2)
        self.parser.add_argument('--distill_with_t_temp', type=float, default=2.0)
        self.parser.add_argument('--distill_with_t_dir', type=str, default='sym', choices=['mlp_to_t','t_to_mlp','sym'])

        # R-GAT
        self.parser.add_argument('--rel_num_bases', type=int, default=3)
        self.parser.add_argument('--rel_use_knn', action='store_true', default=True)
        self.parser.add_argument('--rel_knn_k', type=int, default=5)
        self.parser.add_argument('--rel_knn_min_sim', type=float, default=0.5)
        self.parser.add_argument('--rel_edge_drop', type=float, default=0.1)

    def parse_args(self):
        args = self.parser.parse_args()

        # Number of emotion categories per dataset
        if args.dataset == 'iemocap':
            args.n_emotions = 6 if args.use_emocate else 2
        elif args.dataset == 'meld':
            args.n_emotions = 7 if args.use_emocate else 2
        elif args.dataset == 'dailydialog':
            args.n_emotions = 7 if args.use_emocate else 2

        # Directories
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        return args


def get_model_config(args):
    """Map argparse arguments to ModelConfig"""
    from models import ModelConfig
    config = ModelConfig(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        position_embedding_dim=args.position_embedding_dim,
        max_doc_len=args.max_doc_len,
        n_heads=args.n_heads,
        
        # Node-level embeddings
        use_pos_embed=args.use_pos_embed,
        use_speaker_embed=args.use_speaker_embed,
        speaker_vocab_size=args.speaker_vocab_size,
        
        graph_num_layers=args.graph_num_layers,
        graph_window_size=args.graph_window_size,
        use_speaker_edges=args.use_speaker_edges,
        use_temporal_edges=args.use_temporal_edges,
        distance_decay=args.distance_decay,
        graph_tau=args.graph_tau,

        n_emotions=args.n_emotions,
        use_ot_head=args.use_ot_head,
        fgw_alpha=args.fgw_alpha,
        fgw_eps=args.fgw_eps,
        fgw_iterations=args.fgw_iterations,
        fgw_sinkhorn_iter=args.fgw_sinkhorn_iter,
        fgw_sinkhorn_eps=args.fgw_sinkhorn_eps,
        fgw_row_norm=getattr(args, 'fgw_row_norm', 'entmax'),
        fgw_row_temp=getattr(args, 'fgw_row_temp', 1.0),
        fgw_entmax_alpha=getattr(args, 'fgw_entmax_alpha', 1.5),
        fgw_entmax_bisect_iter=getattr(args, 'fgw_entmax_bisect_iter', 50),
        fgw_use_uot=getattr(args, 'fgw_use_uot', False),
        fgw_uot_rho=getattr(args, 'fgw_uot_rho', 1.0),

        pred_future_cause=args.pred_future_cause,
        train_max_pair_distance=args.train_max_pair_distance,
        mcmf_row_capacity=getattr(args, 'mcmf_row_capacity', 1),
        mcmf_col_capacity=getattr(args, 'mcmf_col_capacity', 1),

        rel_num_bases=getattr(args, 'rel_num_bases', 4),
        rel_use_knn=getattr(args, 'rel_use_knn', True),
        rel_knn_k=getattr(args, 'rel_knn_k', 6),
        rel_knn_min_sim=getattr(args, 'rel_knn_min_sim', 0.5),
        rel_edge_drop=getattr(args, 'rel_edge_drop', 0.1),
    )
    return config


 

