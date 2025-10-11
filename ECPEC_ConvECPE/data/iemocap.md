# IEMOCAP 数据集

## 📖 数据集简介

IEMOCAP (Interactive Emotional Dyadic Motion Capture) 是一个**多模态情感对话数据集**，包含即兴表演和脚本对话，专注于情感原因对抽取（Emotion-Cause Pair Extraction, ECPE）任务。本版本为纯文本 JSON 格式，从原始多模态 H5 文件转换而来。

### 🎯 数据集特点

- **对话来源**: 即兴表演 + 脚本对话
- **对话结构**: 双人对话（男性 M、女性 F）
- **对话长度**: 长对话（平均 50 句）
- **情感密度**: 中高（0.77 对/话语）
- **情感强度**: 高强度情感表达
- **应用场景**: 情感识别、原因抽取、情感强度分析

## 📊 数据统计

### 整体统计

| 分割 | 对话数 | 话语数 | 情感-原因对 | 平均长度 | 文件大小 |
|------|--------|--------|-------------|----------|----------|
| **训练集** (train.json) | 120 | 5,810 | 4,486 | 48.4 | 1.51 MB |
| **测试集** (test.json) | 31 | 1,623 | 1,239 | 52.4 | 0.42 MB |
| **总计** | **151** | **7,433** | **5,725** | **49.2** | **1.93 MB** |

### 关键指标

- **情感-原因对密度**: 0.77 对/话语
- **对话长度范围**: 8-110 句
- **平均对话长度**: 49.2 句（**最长**）
- **说话人数量**: 2 人（M、F）
- **会话划分**: Ses01-Ses04（训练）、Ses05（测试）

## 🎭 情感分布

### 总体情感分布

| 情感类型 | 数量 | 占比 | 描述 |
|---------|------|------|------|
| **frustration** | 1,849 | 24.88% | 挫折、沮丧 |
| **sadness** | 1,708 | 22.98% | 悲伤、难过 |
| **anger** | 1,103 | 14.84% | 愤怒、生气 |
| **happiness** | 1,084 | 14.58% | 快乐、高兴 |
| **fear** | 1,041 | 14.01% | 恐惧、害怕 |
| **neutral** | 648 | 8.72% | 中性情感 |

### 情感分布特点

- ✅ **负面情感主导**: frustration、sadness、anger 占 62.7%
- ✅ **情感强度高**: 包含 frustration（挫折）这一独特情感类型
- ✅ **情感真实**: 即兴表演，情感表达自然强烈
- ⚠️ **neutral 占比低**: 仅 8.72%，与其他数据集差异明显

### 训练集 vs 测试集情感分布

#### 训练集（Ses01-Ses04）
| 情感类型 | 数量 | 占比 |
|---------|------|------|
| frustration | 1,468 | 25.3% |
| sadness | 1,324 | 22.8% |
| anger | 933 | 16.1% |
| happiness | 839 | 14.4% |
| fear | 742 | 12.8% |
| neutral | 504 | 8.7% |

#### 测试集（Ses05）
| 情感类型 | 数量 | 占比 |
|---------|------|------|
| sadness | 384 | 23.7% |
| frustration | 381 | 23.5% |
| fear | 299 | 18.4% |
| happiness | 245 | 15.1% |
| anger | 170 | 10.5% |
| neutral | 144 | 8.9% |

## 👥 说话人分布

| 说话人 | 话语数 | 占比 | 描述 |
|--------|--------|------|------|
| **M** | 3,939 | 52.99% | 男性说话人 |
| **F** | 3,494 | 47.01% | 女性说话人 |

### 说话人特点

- ✅ **双人对话**: 简单的 M-F 标识
- ✅ **分布均衡**: 男女说话人比例接近 1:1
- ✅ **长对话**: 平均每个对话约 50 句，适合研究长程依赖

## 📝 数据格式

### JSON 文件结构

```json
[
  {
    "conversation_ID": 1,
    "conversation": [
      {
        "utterance_ID": 1,
        "text": "Hello?",
        "speaker": "M",
        "emotion": "sadness"
      },
      {
        "utterance_ID": 2,
        "text": "Oh God I finally got...you know how long I've been waiting on line?  You put on hold for like five hours.  Geez.",
        "speaker": "F",
        "emotion": "frustration"
      },
      {
        "utterance_ID": 3,
        "text": "I'm sorry ma'am.  What's the nature of your problem?  But first can I get you to tell me your first and last name?",
        "speaker": "M",
        "emotion": "sadness"
      }
    ],
    "emotion-cause_pairs": [
      [
        "2_frustration",
        "2_Oh God I finally got...you know how long I've been waiting on line?  You put on hold for like five..."
      ],
      [
        "4_frustration",
        "4_I've already been through this five times with these people.  I've been calling every single day."
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
- **speaker**: 说话人标识（M 或 F）
- **emotion**: 话语的情感标签

#### 情感-原因对格式
每个情感-原因对包含两个元素：
1. **情感标签**: `"{utterance_ID}_{emotion}"` 
   - 例如: `"2_frustration"` 表示第 2 句话的情感是 frustration
2. **原因标签**: `"{cause_ID}_{cause_text}"` 
   - 例如: `"2_Oh God I finally got..."` 表示原因是第 2 句话的内容片段

## 💡 使用示例

### 加载数据集

```python
import json

# 加载训练集
with open('iemocap/train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# 加载测试集
with open('iemocap/test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

print(f"训练集: {len(train_data)} 个对话")
print(f"测试集: {len(test_data)} 个对话")
```

### 分析对话长度

```python
# 统计对话长度分布
train_lengths = [len(conv['conversation']) for conv in train_data]
test_lengths = [len(conv['conversation']) for conv in test_data]

print(f"训练集对话长度: {min(train_lengths)}-{max(train_lengths)} (平均 {sum(train_lengths)/len(train_lengths):.1f})")
print(f"测试集对话长度: {min(test_lengths)}-{max(test_lengths)} (平均 {sum(test_lengths)/len(test_lengths):.1f})")
```

### 分析情感分布

```python
from collections import Counter

# 统计情感分布
emotions = Counter()
for conv in train_data:
    for utt in conv['conversation']:
        emotions[utt['emotion']] += 1

print("训练集情感分布:")
for emotion, count in emotions.most_common():
    pct = count / sum(emotions.values()) * 100
    print(f"  {emotion:15s}: {count:5d} ({pct:5.2f}%)")
```

### 提取长对话示例

```python
# 找出最长的对话
longest_conv = max(train_data, key=lambda c: len(c['conversation']))

print(f"\n最长对话 ID: {longest_conv['conversation_ID']}")
print(f"对话长度: {len(longest_conv['conversation'])} 句")
print(f"情感-原因对: {len(longest_conv['emotion-cause_pairs'])} 对")

# 显示前 10 句
print("\n前 10 句话:")
for utt in longest_conv['conversation'][:10]:
    print(f"  {utt['speaker']}: {utt['text'][:60]}... [{utt['emotion']}]")
```

## 📈 数据集示例

### 示例 1: 挫折情感对话（客服场景）

```
对话 ID: 1

M: Hello? [sadness]
F: Oh God I finally got...you know how long I've been waiting on line?  You put on hold for like five hours.  Geez. [frustration]
M: I'm sorry ma'am.  What's the nature of your problem?  But first can I get you to tell me your first and last name? [sadness]
F: I've already been through this five times with these people.  I've been calling every single day. [frustration]
M: I know I just need to verify every time.  I'm sorry. [sadness]
F: Olivia Brown. [frustration]
M: Alright Ms. Brown what's the nature of your problem? [sadness]
F: Well my phone isn't working and it hasn't been working for the last five days. [frustration]
M: Oh.  I'm sorry. Um- Did you drop it in water? [sadness]
F: Yeah, yeah, yeah you better be sorry. [anger]
F: No I did not drop my phone in water.  I'm not five years old and just dropped my phone anywhere I see it.  Okay this is my business. [anger]
F: Okay?  And it [GARBAGE]. My business is failing because my phone isn't working and I don't have a land line and I really can't handle this right now. [frustration]

情感-原因对:
  2_frustration ← 2_Oh God I finally got...you know how long I've been waiting on line?
  4_frustration ← 4_I've already been through this five times with these people.
  6_frustration ← 4_I've already been through this five times with these people.
  ...
```

## 🎯 应用场景

### 1. 长对话情感识别
- 识别长对话中的情感变化
- 研究情感在长对话中的演变

### 2. 高强度情感分析
- 分析 frustration、anger 等强烈情感
- 研究负面情感的触发和传播

### 3. 情感原因对抽取
- 在长对话中抽取情感-原因对
- 处理长程依赖关系

### 4. 即兴对话研究
- 研究自然、即兴的情感表达
- 分析真实情感的产生机制

## 📊 数据集优势

### ✅ 优点

1. **长对话**: 平均 50 句，适合研究长程依赖
2. **情感强度高**: 负面情感占比高，情感表达强烈
3. **独特情感**: 包含 frustration（挫折）情感类型
4. **即兴表演**: 情感表达自然真实
5. **情感密度高**: 0.77 对/话语，信息丰富
6. **会话划分**: 训练/测试集完全分离（不同会话）

### ⚠️ 注意事项

1. **数据量小**: 仅 151 个对话，相比其他数据集较少
2. **对话很长**: 平均 50 句，计算成本高
3. **负面情感多**: 可能不适合某些应用场景
4. **说话人简单**: 仅 M、F 两人，缺少多人对话
5. **纯文本**: 本版本不包含视觉和音频特征

## 🔄 多模态版本

本数据集原始为多模态格式（H5 文件），包含：
- ✅ **文本特征**: 已包含在本 JSON 文件中
- ❌ **音频特征**: 100 维（原 96 维填充），需使用原始 H5 文件
- ❌ **视觉特征**: 100 维，需使用原始 H5 文件

如需多模态特征，请使用原始 H5 文件：
- `iemocap/train.h5` (10.92 MB)
- `iemocap/test.h5` (3.05 MB)

### 多模态特征详情

#### 音频特征
- **维度**: 100 维（原 96 维填充）
- **训练集范围**: [-35.941, 33.081]
- **测试集范围**: [-27.734, 33.538]
- **特征已标准化**

#### 视觉特征
- **维度**: 100 维
- **训练集范围**: [0.000, 0.592]
- **测试集范围**: [0.000, 1.308]
- **特征已标准化**

## 🔍 与其他数据集对比

| 特征 | DailyDialog | MELD | IEMOCAP |
|------|-------------|------|---------|
| **来源** | 日常对话 | 电视剧《老友记》 | 即兴表演 |
| **说话人** | 2 人（A、B） | 多人（6 主角） | 2 人（M、F） |
| **对话长度** | 短（~10 句） | 短（~10 句） | **长（~50 句）** |
| **情感密度** | 高（0.86） | 中（0.72） | 中高（0.77） |
| **情感类型** | 7 种 | 7 种 | **6 种（含 frustration）** |
| **对话数** | 1,106 | 1,351 | **151** |
| **话语数** | 11,104 | 13,352 | 7,433 |
| **特点** | 日常场景 | 幽默多人对话 | **情感强度高** |
| **负面情感** | 低（~10%） | 中（~24%） | **高（~63%）** |

## 📚 引用

如果您使用本数据集，请引用原始论文：

```bibtex
@article{busso2008iemocap,
  title={IEMOCAP: Interactive emotional dyadic motion capture database},
  author={Busso, Carlos and Bulut, Murtaza and Lee, Chi-Chun and Kazemzadeh, Abe and Mower, Emily and Kim, Samuel and Chang, Jeannette N and Lee, Sungbok and Narayanan, Shrikanth S},
  journal={Language resources and evaluation},
  volume={42},
  number={4},
  pages={335--359},
  year={2008},
  publisher={Springer}
}
```

## 📄 许可证

请遵守原数据集的使用条款和许可证要求。

## 🔗 相关资源

- **原始论文**: [IEMOCAP: Interactive emotional dyadic motion capture database](https://link.springer.com/article/10.1007/s10579-008-9076-6)
- **项目主页**: [IEMOCAP Database](https://sail.usc.edu/iemocap/)

## 📌 特别说明

### 会话划分
- **训练集**: Ses01, Ses02, Ses03, Ses04（120 个对话）
- **测试集**: Ses05（31 个对话）
- **无重叠**: 训练集和测试集来自不同会话，完全分离

### 原始话语 ID
原始 H5 文件中包含 `original_utterance_IDs` 字段（如 "Ses03M_impro08b_M000"），本 JSON 版本未包含，如需请使用原始 H5 文件。

---

**最后更新**: 2025-10-05  
**数据格式版本**: JSON v1.0（从 H5 转换）  
**转换工具**: convert_h5_to_json.py

