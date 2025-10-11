# -*- coding: utf-8 -*-

import os

def get_roberta_model_path():
    """获取RoBERTa模型的绝对路径"""
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建roberta文件夹的路径（在父目录下）
    roberta_path = os.path.join(os.path.dirname(current_dir), 'roberta')
    
    # 检查路径是否存在
    if not os.path.exists(roberta_path):
        print(f"警告: RoBERTa模型路径不存在: {roberta_path}")
        print("请确保roberta文件夹位于MECPE_pytorch的同级目录下")
        # 返回相对路径作为后备
        return '../roberta'
    
    return roberta_path

def ensure_model_path_exists(model_path):
    """确保模型路径存在"""
    if model_path.startswith('../roberta') or model_path == '../roberta':
        return get_roberta_model_path()
    return model_path

# 全局模型路径
ROBERTA_MODEL_PATH = get_roberta_model_path()

if __name__ == "__main__":
    print(f"RoBERTa模型路径: {ROBERTA_MODEL_PATH}")
    print(f"路径是否存在: {os.path.exists(ROBERTA_MODEL_PATH)}")

