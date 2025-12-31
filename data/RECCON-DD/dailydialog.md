# DailyDialog 数据集

## 📖 数据集简介

DailyDialog 是一个高质量的**日常对话数据集**，专注于情感原因对抽取（Emotion-Cause Pair Extraction, ECPE）任务。数据集包含日常生活中各种场景的双人对话，涵盖多种情感类型。

### 🎯 数据集特点

- **对话类型**: 日常生活对话
- **对话结构**: 简单的 A-B 双人对话
- **对话长度**: 短对话（平均 10 句）
- **情感密度**: 高（0.86 对/话语）
- **标注质量**: 高质量人工标注
- **应用场景**: 情感识别、原因抽取、对话理解

## 📊 数据统计

### 整体统计

| 分割 | 对话数 | 话语数 | 情感-原因对 | 平均长度 |
|------|--------|--------|-------------|----------|
| **训练集** (train.json) | 834 | 8,206 | 7,271 | 9.8 |
| **验证集** (vaild.json) | 47 | 493 | 347 | 10.5 |
| **测试集** (test.json) | 225 | 2,405 | 1,894 | 10.7 |
| **总计** | **1,106** | **11,104** | **9,512** | **10.0** |

### 关键指标

- **情感-原因对密度**: 0.86 对/话语
- **对话长度范围**: 3-31 句
- **平均对话长度**: 10.0 句
- **说话人数量**: 2 人（A、B）

## 🎭 情感分布

### 总体情感分布

| 情感类型 | 数量 | 占比 | 描述 |
|---------|------|------|------|
| **neutral** | 5,243 | 47.22% | 中性情感 |
| **happiness** | 4,361 | 39.27% | 快乐、高兴 |
| **surprise** | 484 | 4.36% | 惊讶 |
| **anger** | 451 | 4.06% | 愤怒、生气 |
| **sadness** | 351 | 3.16% | 悲伤、难过 |
| **disgust** | 140 | 1.26% | 厌恶、反感 |
| **fear** | 74 | 0.67% | 恐惧、害怕 |

### 情感分布特点

- ✅ **情感平衡**: neutral 和 happiness 占主导（86.5%），符合日常对话特点
- ✅ **情感多样**: 涵盖 7 种基本情感类型
- ✅ **真实性高**: 情感分布符合日常生活场景

## 👥 说话人分布

| 说话人 | 话语数 | 占比 |
|--------|--------|------|
| **A** | 5,762 | 51.89% |
| **B** | 5,342 | 48.11% |

- 双人对话，说话人分布均衡
- 简单的 A-B 标识，便于处理

## 📝 数据格式

### JSON 文件结构

```json
[
  {
    "conversation_ID": 1,
    "conversation": [
      {
        "utterance_ID": 1,
        "text": "Hey , you wanna see a movie tomorrow ?",
        "speaker": "A",
        "emotion": "happiness"
      },
      {
        "utterance_ID": 2,
        "text": "Sounds like a good plan . What do you want to see ?",
        "speaker": "B",
        "emotion": "happiness"
      }
    ],
    "emotion-cause_pairs": [
      [
        "1_happiness",
        "1_see a movie tomorrow ?"
      ],
      [
        "2_happiness",
        "1_see a movie tomorrow ?"
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
- **speaker**: 说话人标识（A 或 B）
- **emotion**: 话语的情感标签

#### 情感-原因对格式
每个情感-原因对包含两个元素：
1. **情感标签**: `"{utterance_ID}_{emotion}"` 
   - 例如: `"1_happiness"` 表示第 1 句话的情感是 happiness
2. **原因标签**: `"{cause_ID}_{cause_text}"` 
   - 例如: `"1_see a movie tomorrow ?"` 表示原因是第 1 句话的内容片段

## 💡 使用示例

### 加载数据集

```python
import json

# 加载训练集
with open('dailydialog/train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

print(f"训练集包含 {len(train_data)} 个对话")
```

### 遍历对话

```python
for conv in train_data:
    conv_id = conv['conversation_ID']
    print(f"\n对话 {conv_id}:")
    
    # 遍历话语
    for utt in conv['conversation']:
        print(f"  {utt['speaker']}: {utt['text']} [{utt['emotion']}]")
    
    # 遍历情感-原因对
    print(f"\n  情感-原因对:")
    for emotion_label, cause_label in conv['emotion-cause_pairs']:
        print(f"    {emotion_label} ← {cause_label}")
```

### 统计情感分布

```python
from collections import Counter

emotions = Counter()
for conv in train_data:
    for utt in conv['conversation']:
        emotions[utt['emotion']] += 1

print("情感分布:")
for emotion, count in emotions.most_common():
    print(f"  {emotion}: {count}")
```

## 📈 数据集示例

### 示例 1: 快乐情感对话

```
对话 ID: 1

A: Hey , you wanna see a movie tomorrow ? [happiness]
B: Sounds like a good plan . What do you want to see ? [happiness]
A: How about Legally Blonde . [neutral]
B: Ah , my girlfriend wanted to see that movie . I have to take her later so I don't want to watch it ahead of time . How about The Cube ? [neutral]
A: Isn't that a scary movie ? [neutral]
B: How scary can it be ? Come on , it'll be fun . [neutral]
A: Ok . I'll give it a try . [happiness]
B: That's the spirit . I'll see you tomorrow after class . [happiness]
A: Ok . See you tomorrow . [happiness]

情感-原因对:
  1_happiness ← 1_see a movie tomorrow ?
  2_happiness ← 1_see a movie tomorrow ?
  7_happiness ← 6_it'll be fun .
  8_happiness ← 6_it'll be fun .
  8_happiness ← 7_Ok . I'll give it a try .
  9_happiness ← 1_see a movie tomorrow ?
  9_happiness ← 6_it'll be fun .
```

### 示例 2: 愤怒情感对话

```
对话 ID: 2

A: Ann , it's terrible ! [anger]
B: What's up ? [neutral]
A: Look , this is a pimple ! [neutral]
B: Oh , I think it is . [neutral]
A: How come ? [neutral]
B: I think it's because of your bad habits ! [anger]
A: I have no bad habit . I sleep eight hours a day , never eat spicy food , clean the face twice a day and so on . I have done a lot . [neutral]
B: I know you have done a lot , but you always sleep very late . Sleeping eight hours a day doesn't mean it is healthy . Sleeping after 12 is hurtful to our body , and I think this is your problem . [neutral]
A: I wasn't aware of that ! [surprise]
B: You should do better later . [neutral]
```

## 🎯 应用场景

### 1. 情感识别
- 识别对话中每句话的情感类型
- 多分类任务（7 种情感）

### 2. 情感原因对抽取
- 识别情感话语及其原因话语
- 理解情感产生的上下文

### 3. 对话理解
- 分析对话流程和情感变化
- 研究情感在对话中的传播

### 4. 情感生成
- 生成带有特定情感的对话
- 控制对话的情感走向

## 📊 数据集优势

### ✅ 优点

1. **高质量标注**: 人工标注，准确度高
2. **情感密度高**: 平均 0.86 对/话语，信息丰富
3. **场景真实**: 涵盖日常生活各种场景
4. **结构简单**: 双人对话，易于处理
5. **情感多样**: 7 种情感类型，覆盖全面
6. **长度适中**: 平均 10 句，适合模型训练

### ⚠️ 注意事项

1. **说话人简单**: 仅 A、B 两人，缺少多人对话场景
2. **对话较短**: 平均 10 句，不适合长对话研究
3. **情感不均衡**: neutral 和 happiness 占比较高
4. **纯文本**: 无多模态信息（视觉、音频）

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

## 📚 引用

如果您使用本数据集，请引用原始论文：

```bibtex
@article{dailydialog,
  title={DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset},
  author={Li, Yanran and Su, Hui and Shen, Xiaoyu and Li, Wenjie and Cao, Ziqiang and Niu, Shuzi},
  journal={arXiv preprint arXiv:1710.03957},
  year={2017}
}
```

## 📄 许可证

请遵守原数据集的使用条款和许可证要求。

## 🔗 相关资源

- **原始论文**: [DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset](https://arxiv.org/abs/1710.03957)
- **项目主页**: 请参考原始数据集发布页面

---

**最后更新**: 2025-10-05  
**数据格式版本**: JSON v1.0

