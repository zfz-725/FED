#!/usr/bin/env python3
"""
生成中文PSI测试数据
包含：
1. 精确匹配
2. 模糊匹配（不同相似度级别）
3. 部分匹配（包含关系）
4. 语义相似但表述不同
5. 长文本匹配
6. 特殊字符和格式
"""

import os
import random

# 创建测试目录
os.makedirs('test', exist_ok=True)

# 中文字符集
CHINESE_CHARS = '的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面方后多定行学义自女力线本水理化社党制政军事情己道变通合质量级标统林营经活济基本制级标统林营经活济基本级标统林营经活济基本'

# 常用中文词汇
CHINESE_WORDS = [
    '机器学习', '人工智能', '深度学习', '数据隐私', '联邦学习',
    '神经网络', '自然语言处理', '计算机视觉', '数据挖掘', '算法',
    '隐私保护', '安全计算', '分布式系统', '云计算', '边缘计算',
    '区块链', '大数据', '物联网', '数据安全', '信息保护',
    '梯度下降', '卷积网络', '循环网络', '注意力机制', 'Transformer',
    '数据去重', '哈希函数', '局部敏感哈希', '相似度计算', '模糊匹配'
]

# 中文句子模板
CHINESE_SENTENCES = [
    '{}是一个重要的研究领域',
    '{}技术正在改变我们的生活',
    '{}能够有效解决{}问题',
    '{}的发展前景非常广阔',
    '{}在{}中发挥着重要作用',
    '{}需要考虑{}和{}的平衡',
    '{}的核心技术包括{}和{}',
    '{}已经在{}领域取得了显著成果',
    '{}面临的挑战包括{}和{}',
    '{}的未来发展方向是{}'
]

def generate_random_chinese(length):
    """生成随机中文字符串"""
    return ''.join(random.choice(CHINESE_CHARS) for _ in range(length))

def generate_chinese_sentence():
    """生成中文句子"""
    template = random.choice(CHINESE_SENTENCES)
    # 根据模板中的{}数量选择相应数量的词汇
    num_placeholders = template.count('{}')
    words = random.sample(CHINESE_WORDS, num_placeholders)
    return template.format(*words)

def generate_similar_chinese(base_text, similarity_ratio):
    """
    基于基础文本生成相似的中文文本
    similarity_ratio: 0-1之间的值，越高越相似
    """
    if similarity_ratio >= 0.9:
        # 轻微修改：替换1-2个字符
        chars = list(base_text)
        num_changes = max(1, int(len(chars) * 0.05))
        for _ in range(num_changes):
            pos = random.randint(0, len(chars) - 1)
            if chars[pos] in CHINESE_CHARS:
                chars[pos] = random.choice(CHINESE_CHARS)
        return ''.join(chars)
    
    elif similarity_ratio >= 0.8:
        # 中等修改：替换一些词汇
        words = base_text.split()
        num_words = len(words)
        num_changes = max(1, int(num_words * 0.2))
        modified_words = words.copy()
        for _ in range(num_changes):
            pos = random.randint(0, num_words - 1)
            modified_words[pos] = random.choice(CHINESE_WORDS)
        return ' '.join(modified_words)
    
    elif similarity_ratio >= 0.7:
        # 较大修改：删除和添加词汇
        words = base_text.split()
        modified_words = words.copy()
        # 删除一些词
        num_delete = max(1, int(len(modified_words) * 0.15))
        for _ in range(num_delete):
            if len(modified_words) > 3:
                pos = random.randint(0, len(modified_words) - 1)
                del modified_words[pos]
        # 添加一些词
        num_add = max(1, int(len(words) * 0.15))
        for _ in range(num_add):
            pos = random.randint(0, len(modified_words))
            modified_words.insert(pos, random.choice(CHINESE_WORDS))
        return ' '.join(modified_words)
    
    else:
        # 大幅修改：重新生成句子
        return generate_chinese_sentence()

def generate_party_a_data():
    """生成参与方A的中文测试数据"""
    data = []
    
    # 1. 精确匹配组（5条）
    exact_matches = [
        "机器学习是人工智能的核心技术",
        "隐私保护在联邦学习中至关重要",
        "数据去重能够有效减少存储成本",
        "神经网络已经彻底改变了人工智能",
        "自然语言处理让计算机理解人类语言"
    ]
    data.extend(exact_matches)
    
    # 2. 高相似度模糊匹配（相似度0.9+，5条）
    high_similarity_bases = [
        "联邦学习实现了多方协作模型训练",
        "安全多方计算保护数据隐私",
        "梯度下降优化最小化损失函数",
        "卷积神经网络在图像识别方面表现出色",
        "自然语言处理理解人类交流"
    ]
    for base in high_similarity_bases:
        data.append(generate_similar_chinese(base, 0.92))
    
    # 3. 中等相似度模糊匹配（相似度0.8+，5条）
    medium_similarity_bases = [
        "分布式系统处理大规模数据",
        "加密算法保护敏感信息",
        "深度学习模型需要大量计算资源",
        "数据预处理提高模型性能",
        "特征提取识别相关模式"
    ]
    for base in medium_similarity_bases:
        data.append(generate_similar_chinese(base, 0.85))
    
    # 4. 低相似度模糊匹配（相似度0.7+，5条）
    low_similarity_bases = [
        "区块链技术确保数据完整性",
        "云计算提供可扩展基础设施",
        "边缘计算减少网络延迟",
        "物联网连接物理设备",
        "大数据分析提取有价值见解"
    ]
    for base in low_similarity_bases:
        data.append(generate_similar_chinese(base, 0.75))
    
    # 5. 长文本（3条，每条约100-200字）
    long_texts = [
        "在机器学习领域，联邦学习已经成为一种有前途的方法，它使多个参与方能够协作训练共享模型，而无需共享原始数据。这种范式解决了关键的隐私问题，同时有效地利用了分布式数据源。",
        "数据隐私法规如GDPR和CCPA已经显著影响了组织处理个人信息的方式。合规要求实施强大的技术保障措施和透明的数据治理实践。",
        "深度学习架构的进步，包括Transformer和注意力机制，已经在自然语言理解方面取得了突破。这些模型现在可以执行复杂的任务，如翻译、摘要和问答，准确度令人印象深刻。"
    ]
    data.extend(long_texts)
    
    # 6. 包含特殊字符和格式的文本（5条）
    special_format_texts = [
        "错误代码：404 - 未找到文件路径：/路径/到/资源",
        "用户ID：12345 | 时间戳：2024-01-15 14:30:00 | 操作：登录",
        "函数计算(x, y) { return x * y + Math.sqrt(x); }",
        "SELECT * FROM users WHERE age > 18 AND status = 'active';",
        "API响应：{\"status\": \"success\", \"data\": [1, 2, 3, 4, 5]}"
    ]
    data.extend(special_format_texts)
    
    # 7. 完全不同文本（5条，用于对比）
    different_texts = [
        generate_random_chinese(30),
        generate_random_chinese(40),
        generate_random_chinese(35),
        generate_random_chinese(45),
        generate_random_chinese(25)
    ]
    data.extend(different_texts)
    
    return data

def generate_party_b_data():
    """生成参与方B的中文测试数据"""
    data = []
    
    # 1. 与A的精确匹配（5条）
    exact_matches = [
        "机器学习是人工智能的核心技术",
        "隐私保护在联邦学习中至关重要",
        "数据去重能够有效减少存储成本",
        "神经网络已经彻底改变了人工智能",
        "自然语言处理让计算机理解人类语言"
    ]
    data.extend(exact_matches)
    
    # 2. 与A的高相似度文本对应（相似度0.9+，5条）
    high_similarity_bases = [
        "联邦学习实现了多方协作模型训练",
        "安全多方计算保护数据隐私",
        "梯度下降优化最小化损失函数",
        "卷积神经网络在图像识别方面表现出色",
        "自然语言处理理解人类交流"
    ]
    for base in high_similarity_bases:
        # 生成与A不同的变体，但同样相似
        data.append(generate_similar_chinese(base, 0.93))
    
    # 3. 与A的中等相似度文本对应（相似度0.8+，5条）
    medium_similarity_bases = [
        "分布式系统处理大规模数据",
        "加密算法保护敏感信息",
        "深度学习模型需要大量计算资源",
        "数据预处理提高模型性能",
        "特征提取识别相关模式"
    ]
    for base in medium_similarity_bases:
        data.append(generate_similar_chinese(base, 0.82))
    
    # 4. 与A的低相似度文本对应（相似度0.7+，5条）
    low_similarity_bases = [
        "区块链技术确保数据完整性",
        "云计算提供可扩展基础设施",
        "边缘计算减少网络延迟",
        "物联网连接物理设备",
        "大数据分析提取有价值见解"
    ]
    for base in low_similarity_bases:
        data.append(generate_similar_chinese(base, 0.78))
    
    # 5. 长文本（3条，与A的长文本相似但不完全相同）
    long_texts = [
        "联邦学习代表了机器学习的重大进步，允许各种组织在保持数据本地化的同时共同构建共享模型。这种方法有效解决了隐私问题，并充分利用了多样化的数据源。",
        "世界各地的组织现在必须应对像GDPR和CCPA这样复杂的数据隐私法律。满足这些标准需要在整个公司实施强大的技术保护措施和明确的数据管理政策。",
        "深度学习的最新进展，特别是Transformer模型和基于注意力的架构，已经极大地提高了机器理解语言的能力。今天的模型可以翻译语言、创建摘要和回答问题，精确度令人印象深刻。"
    ]
    data.extend(long_texts)
    
    # 6. 包含特殊字符的文本（5条，部分与A相似）
    special_format_texts = [
        "错误代码：404 - 未找到文件路径：/路径/到/资源",
        "时间戳：2024-01-15 14:30:00 | 用户ID：12345 | 操作：登录成功",
        "函数计算(a, b) { return a * b + Math.sqrt(a); }",
        "SELECT id, name FROM users WHERE age >= 18 AND status = 'active';",
        "响应：{\"status\": \"success\", \"count\": 5, \"data\": [1, 2, 3, 4, 5]}"
    ]
    data.extend(special_format_texts)
    
    # 7. 完全不同文本（5条，与A的不同）
    different_texts = [
        generate_random_chinese(32),
        generate_random_chinese(38),
        generate_random_chinese(36),
        generate_random_chinese(42),
        generate_random_chinese(28)
    ]
    data.extend(different_texts)
    
    return data

def main():
    # 生成参与方A的数据
    party_a_data = generate_party_a_data()
    with open('test/party_a_chinese.txt', 'w', encoding='utf-8') as f:
        for item in party_a_data:
            f.write(item + '\n')
    
    # 生成参与方B的数据
    party_b_data = generate_party_b_data()
    with open('test/party_b_chinese.txt', 'w', encoding='utf-8') as f:
        for item in party_b_data:
            f.write(item + '\n')
    
    print("中文测试数据生成成功！")
    print(f"参与方A数据: {len(party_a_data)} 条记录 -> test/party_a_chinese.txt")
    print(f"参与方B数据: {len(party_b_data)} 条记录 -> test/party_b_chinese.txt")
    print("\n数据分布:")
    print("- 精确匹配: 5条")
    print("- 高相似度模糊匹配 (0.9+): 5条")
    print("- 中等相似度模糊匹配 (0.8+): 5条")
    print("- 低相似度模糊匹配 (0.7+): 5条")
    print("- 长文本: 3条")
    print("- 特殊格式文本: 5条")
    print("- 完全不同文本: 5条")

if __name__ == "__main__":
    main()
