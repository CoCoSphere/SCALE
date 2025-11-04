# MELD 数据集

## 📖 数据集简介

MELD (Multimodal EmotionLines Dataset) 是一个基于电视剧《老友记》(Friends) 的**多模态情感对话数据集**，专注于情感原因对抽取（Emotion-Cause Pair Extraction, ECPE）任务。本版本为纯文本 JSON 格式，从原始多模态 H5 文件转换而来。

### 🎯 数据集特点

- **对话来源**: 电视剧《老友记》
- **对话结构**: 多人对话（6 位主角 + 配角）
- **对话长度**: 短对话（平均 10 句）
- **情感密度**: 中等（0.72 对/话语）
- **标注质量**: 高质量人工标注
- **应用场景**: 情感识别、原因抽取、多人对话理解

## 📊 数据统计

### 整体统计

| 分割 | 对话数 | 话语数 | 情感-原因对 | 平均长度 | 文件大小 |
|------|--------|--------|-------------|----------|----------|
| **训练集** (train.json) | 984 | 9,764 | 6,896 | 9.9 | 2.37 MB |
| **验证集** (dev.json) | 110 | 1,069 | 838 | 9.7 | 0.27 MB |
| **测试集** (test.json) | 257 | 2,519 | 1,828 | 9.8 | 0.62 MB |
| **总计** | **1,351** | **13,352** | **9,562** | **9.9** | **3.26 MB** |

### 关键指标

- **情感-原因对密度**: 0.72 对/话语
- **对话长度范围**: 1-33 句
- **平均对话长度**: 9.9 句
- **说话人数量**: 主要 6 人 + 多位配角

## 🎭 情感分布

### 总体情感分布

| 情感类型 | 数量 | 占比 | 描述 |
|---------|------|------|------|
| **neutral** | 5,808 | 43.50% | 中性情感 |
| **joy** | 2,258 | 16.91% | 快乐、喜悦 |
| **surprise** | 1,787 | 13.38% | 惊讶 |
| **anger** | 1,584 | 11.86% | 愤怒、生气 |
| **sadness** | 1,140 | 8.54% | 悲伤、难过 |
| **disgust** | 403 | 3.02% | 厌恶、反感 |
| **fear** | 372 | 2.79% | 恐惧、害怕 |

### 情感分布特点

- ✅ **情感丰富**: 涵盖 7 种基本情感类型
- ✅ **幽默场景多**: surprise 和 joy 占比较高（30.3%），符合情景喜剧特点
- ✅ **情感多样**: 相比 DailyDialog，负面情感（anger, sadness）占比更高

## 👥 说话人分布

### 主要角色（前 10 名）

| 说话人 | 话语数 | 占比 | 角色描述 |
|--------|--------|------|----------|
| **Joey** | 2,020 | 15.13% | 乔伊·崔比雅尼 |
| **Ross** | 1,982 | 14.84% | 罗斯·盖勒 |
| **Rachel** | 1,918 | 14.36% | 瑞秋·格林 |
| **Phoebe** | 1,727 | 12.93% | 菲比·布菲 |
| **Monica** | 1,721 | 12.89% | 莫妮卡·盖勒 |
| **Chandler** | 1,708 | 12.79% | 钱德勒·宾 |
| **Janice** | 84 | 0.63% | 珍妮丝（配角） |
| **Tag** | 60 | 0.45% | 泰格（配角） |
| **Emily** | 59 | 0.44% | 艾米丽（配角） |
| **Carol** | 54 | 0.40% | 卡萝（配角） |

### 说话人特点

- ✅ **6 位主角**: 话语分布均衡（12.8%-15.1%）
- ✅ **多位配角**: 增加对话的真实性和多样性
- ✅ **多人对话**: 适合研究多人交互场景

## 📝 数据格式

### JSON 文件结构

```json
[
  {
    "conversation_ID": 1,
    "conversation": [
      {
        "utterance_ID": 1,
        "text": "Alright , so I am back in high school , I am standing in the middle of the cafeteria , and I realize I am totally naked .",
        "speaker": "Chandler",
        "emotion": "neutral"
      },
      {
        "utterance_ID": 2,
        "text": "Oh , yeah . Had that dream .",
        "speaker": "All",
        "emotion": "neutral"
      },
      {
        "utterance_ID": 3,
        "text": "Then I look down , and I realize there is a phone ... there .",
        "speaker": "Chandler",
        "emotion": "surprise"
      }
    ],
    "emotion-cause_pairs": [
      [
        "3_surprise",
        "1_Alright , so I am back in high school , I am standing in the middle of the cafeteria , and I realize..."
      ],
      [
        "3_surprise",
        "3_Then I look down , and I realize there is a phone ... there ."
      ]
    ]
  }
]
```

### 字段说明

#### 对话级别
- **conversation_ID**: 对话的唯一标识符（整数）
- **conversation**: 对话中的所有话语列表
- **emotion-cause_pairs**: 情感-原因对列表

#### 话语级别
- **utterance_ID**: 话语在对话中的序号（从 1 开始）
- **text**: 话语的文本内容
- **speaker**: 说话人姓名（如 Chandler, Monica, Ross 等）
- **emotion**: 话语的情感标签

#### 情感-原因对格式
每个情感-原因对包含两个元素：
1. **情感标签**: `"{utterance_ID}_{emotion}"` 
   - 例如: `"3_surprise"` 表示第 3 句话的情感是 surprise
2. **原因标签**: `"{cause_ID}_{cause_text}"` 
   - 例如: `"1_Alright , so I am back in high school..."` 表示原因是第 1 句话的内容片段

## 💡 使用示例

### 加载数据集

```python
import json

# 加载训练集
with open('meld/train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

print(f"训练集包含 {len(train_data)} 个对话")
```

### 遍历对话

```python
for conv in train_data[:5]:  # 只显示前 5 个对话
    conv_id = conv['conversation_ID']
    print(f"\n对话 {conv_id}:")
    
    # 遍历话语
    for utt in conv['conversation']:
        print(f"  {utt['speaker']:10s}: {utt['text'][:50]}... [{utt['emotion']}]")
    
    # 统计情感-原因对
    print(f"  情感-原因对数量: {len(conv['emotion-cause_pairs'])}")
```

### 统计说话人分布

```python
from collections import Counter

speakers = Counter()
for conv in train_data:
    for utt in conv['conversation']:
        speakers[utt['speaker']] += 1

print("说话人分布（前 10 名）:")
for speaker, count in speakers.most_common(10):
    print(f"  {speaker:15s}: {count:5d}")
```

### 分析情感-原因对

```python
# 提取情感-原因对示例
for conv in train_data[:3]:
    if not conv['emotion-cause_pairs']:
        continue
    
    print(f"\n对话 {conv['conversation_ID']}:")
    
    # 创建话语映射
    utt_map = {utt['utterance_ID']: utt for utt in conv['conversation']}
    
    for emotion_label, cause_label in conv['emotion-cause_pairs'][:2]:
        emotion_id, emotion_type = emotion_label.split('_', 1)
        cause_id = int(cause_label.split('_', 1)[0])
        
        if int(emotion_id) in utt_map and cause_id in utt_map:
            emotion_utt = utt_map[int(emotion_id)]
            cause_utt = utt_map[cause_id]
            
            print(f"  情感: {emotion_utt['speaker']} - {emotion_type}")
            print(f"    话语: {emotion_utt['text'][:60]}...")
            print(f"  原因: {cause_utt['speaker']}")
            print(f"    话语: {cause_utt['text'][:60]}...")
            print()
```

## 📈 数据集示例

### 示例 1: 惊讶情感对话

```
对话 ID: 1

Chandler: Alright , so I am back in high school , I am standing in the middle of the cafeteria , and I realize I am totally naked . [neutral]
All: Oh , yeah . Had that dream . [neutral]
Chandler: Then I look down , and I realize there is a phone ... there . [surprise]
Joey: Instead of ... ? [surprise]
Chandler: That is right . [anger]
Joey: Never had that dream . [neutral]
Phoebe: No . [neutral]
Chandler: All of a sudden , the phone starts to ring . [neutral]

情感-原因对:
  3_surprise ← 1_Alright , so I am back in high school...
  3_surprise ← 3_Then I look down , and I realize there is a phone ... there .
  4_surprise ← 1_Alright , so I am back in high school...
  4_surprise ← 3_Then I look down , and I realize there is a phone ... there .
  4_surprise ← 4_Instead of ... ?
  5_anger ← 1_Alright , so I am back in high school...
```

## 🎯 应用场景

### 1. 多人对话情感识别
- 识别多人对话中每句话的情感
- 理解不同角色的情感表达

### 2. 情感原因对抽取
- 在多人对话场景下抽取情感-原因对
- 跨说话人的情感原因分析

### 3. 情景喜剧分析
- 研究幽默对话的情感模式
- 分析惊讶和快乐情感的触发机制

### 4. 角色情感建模
- 为不同角色建立情感模型
- 研究角色性格与情感表达的关系

## 📊 数据集优势

### ✅ 优点

1. **多人对话**: 6 位主角 + 配角，真实多人交互场景
2. **情感丰富**: 幽默场景多，surprise 和 joy 占比高
3. **数据量大**: 1,351 个对话，13,352 句话语
4. **角色多样**: 不同性格的角色，情感表达多样
5. **场景真实**: 来自真实电视剧，对话自然流畅
6. **标注详细**: 高质量的情感和原因标注

### ⚠️ 注意事项

1. **领域特定**: 来自情景喜剧，可能不适用于其他领域
2. **文化背景**: 美国文化背景，某些幽默可能难以理解
3. **情感密度**: 相比 DailyDialog 较低（0.72 vs 0.86）
4. **纯文本**: 本版本不包含视觉和音频特征（原始 H5 文件包含）

## 🔄 多模态版本

本数据集原始为多模态格式（H5 文件），包含：
- ✅ **文本特征**: 已包含在本 JSON 文件中
- ❌ **视觉特征**: 4096 维，需使用原始 H5 文件
- ❌ **音频特征**: 6373 维，需使用原始 H5 文件

如需多模态特征，请使用原始 H5 文件：
- `meld/train.h5` (783.8 MB)
- `meld/dev.h5` (85.5 MB)
- `meld/test.h5` (202.4 MB)

## 🔍 与其他数据集对比

| 特征 | DailyDialog | MELD | IEMOCAP |
|------|-------------|------|---------|
| **来源** | 日常对话 | 电视剧《老友记》 | 即兴表演 |
| **说话人** | 2 人（A、B） | 多人（6 主角） | 2 人（M、F） |
| **对话长度** | 短（~10 句） | 短（~10 句） | 长（~50 句） |
| **情感密度** | 高（0.86） | 中（0.72） | 中高（0.77） |
| **情感类型** | 7 种 | 7 种 | 6 种 |
| **对话数** | 1,106 | 1,351 | 151 |
| **话语数** | 11,104 | 13,352 | 7,433 |
| **特点** | 日常场景 | 幽默多人对话 | 情感强度高 |

## 📚 引用

如果您使用本数据集，请引用原始论文：

```bibtex
@inproceedings{poria2019meld,
  title={MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations},
  author={Poria, Soujanya and Hazarika, Devamanyu and Majumder, Navonil and Naik, Gautam and Cambria, Erik and Mihalcea, Rada},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  pages={527--536},
  year={2019}
}
```

## 📄 许可证

请遵守原数据集的使用条款和许可证要求。

## 🔗 相关资源

- **原始论文**: [MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations](https://aclanthology.org/P19-1050/)
- **项目主页**: [MELD Dataset](https://affective-meld.github.io/)
- **GitHub**: [MELD Repository](https://github.com/declare-lab/MELD)

---

**最后更新**: 2025-10-05  
**数据格式版本**: JSON v1.0（从 H5 转换）  
**转换工具**: convert_h5_to_json.py

