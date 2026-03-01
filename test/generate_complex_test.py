#!/usr/bin/env python3
"""
生成复杂的PSI测试数据
包含：
1. 精确匹配
2. 模糊匹配（不同相似度级别）
3. 部分匹配（包含关系）
4. 语义相似但表述不同
5. 长文本匹配
6. 多语言文本
7. 特殊字符和格式
"""

import os
import random
import string

# 创建测试目录
os.makedirs('test', exist_ok=True)

def generate_random_string(length):
    """生成随机字符串"""
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for i in range(length))

def generate_similar_text(base_text, similarity_ratio):
    """
    基于基础文本生成相似文本
    similarity_ratio: 0-1之间的值，越高越相似
    """
    words = base_text.split()
    num_words = len(words)
    
    # 根据相似度决定修改程度
    if similarity_ratio >= 0.9:
        # 轻微修改：替换1-2个字符
        chars = list(base_text)
        num_changes = max(1, int(len(chars) * 0.05))
        for _ in range(num_changes):
            pos = random.randint(0, len(chars) - 1)
            if chars[pos] != ' ':
                chars[pos] = random.choice(string.ascii_lowercase)
        return ''.join(chars)
    
    elif similarity_ratio >= 0.8:
        # 中等修改：替换一些词
        num_changes = max(1, int(num_words * 0.2))
        modified_words = words.copy()
        for _ in range(num_changes):
            pos = random.randint(0, num_words - 1)
            modified_words[pos] = generate_random_string(len(modified_words[pos]))
        return ' '.join(modified_words)
    
    elif similarity_ratio >= 0.7:
        # 较大修改：删除和添加词
        modified_words = words.copy()
        # 删除一些词
        num_delete = max(1, int(num_words * 0.15))
        for _ in range(num_delete):
            if len(modified_words) > 3:
                pos = random.randint(0, len(modified_words) - 1)
                del modified_words[pos]
        # 添加一些词
        num_add = max(1, int(num_words * 0.15))
        for _ in range(num_add):
            pos = random.randint(0, len(modified_words))
            modified_words.insert(pos, generate_random_string(random.randint(3, 8)))
        return ' '.join(modified_words)
    
    else:
        # 大幅修改：重新组织句子结构
        random.shuffle(words)
        return ' '.join(words)

def generate_party_a_data():
    """生成参与方A的复杂测试数据"""
    data = []
    
    # 1. 精确匹配组（5条）
    exact_matches = [
        "the quick brown fox jumps over the lazy dog",
        "machine learning is transforming the way we analyze data",
        "privacy preservation is crucial in federated learning systems",
        "data deduplication helps reduce storage costs significantly",
        "neural networks have revolutionized artificial intelligence"
    ]
    data.extend(exact_matches)
    
    # 2. 高相似度模糊匹配（相似度0.9+，5条）
    high_similarity_bases = [
        "federated learning enables collaborative model training",
        "secure multi-party computation protects data privacy",
        "gradient descent optimization minimizes loss functions",
        "convolutional neural networks excel at image recognition",
        "natural language processing understands human communication"
    ]
    for base in high_similarity_bases:
        data.append(generate_similar_text(base, 0.92))
    
    # 3. 中等相似度模糊匹配（相似度0.8+，5条）
    medium_similarity_bases = [
        "distributed systems handle large scale data processing",
        "encryption algorithms secure sensitive information",
        "deep learning models require substantial computational resources",
        "data preprocessing improves model performance",
        "feature extraction identifies relevant patterns"
    ]
    for base in medium_similarity_bases:
        data.append(generate_similar_text(base, 0.85))
    
    # 4. 低相似度模糊匹配（相似度0.7+，5条）
    low_similarity_bases = [
        "blockchain technology ensures data integrity",
        "cloud computing provides scalable infrastructure",
        "edge computing reduces network latency",
        "internet of things connects physical devices",
        "big data analytics extracts valuable insights"
    ]
    for base in low_similarity_bases:
        data.append(generate_similar_text(base, 0.75))
    
    # 5. 长文本（3条，每条约200-300字符）
    long_texts = [
        "In the field of machine learning, federated learning has emerged as a promising approach that enables multiple parties to collaboratively train a shared model without sharing their raw data. This paradigm addresses critical privacy concerns while leveraging distributed data sources effectively.",
        "Data privacy regulations such as GDPR and CCPA have significantly impacted how organizations handle personal information. Compliance requires implementing robust technical safeguards and transparent data governance practices across all business operations.",
        "The advancement of deep learning architectures, including transformers and attention mechanisms, has led to breakthroughs in natural language understanding. These models can now perform complex tasks like translation, summarization, and question answering with remarkable accuracy."
    ]
    data.extend(long_texts)
    
    # 6. 包含特殊字符和格式的文本（5条）
    special_format_texts = [
        "Error code: 0x404 - File not found at /path/to/resource",
        "User ID: 12345 | Timestamp: 2024-01-15 14:30:00 | Action: LOGIN",
        "function compute(x, y) { return x * y + Math.sqrt(x); }",
        "SELECT * FROM users WHERE age > 18 AND status = 'active';",
        "API Response: {\"status\": \"success\", \"data\": [1, 2, 3, 4, 5]}"
    ]
    data.extend(special_format_texts)
    
    # 7. 完全不同文本（5条，用于对比）
    different_texts = [
        generate_random_string(50),
        generate_random_string(60),
        generate_random_string(45),
        generate_random_string(55),
        generate_random_string(40)
    ]
    data.extend(different_texts)
    
    return data

def generate_party_b_data():
    """生成参与方B的复杂测试数据"""
    data = []
    
    # 1. 与A的精确匹配（5条）
    exact_matches = [
        "the quick brown fox jumps over the lazy dog",
        "machine learning is transforming the way we analyze data",
        "privacy preservation is crucial in federated learning systems",
        "data deduplication helps reduce storage costs significantly",
        "neural networks have revolutionized artificial intelligence"
    ]
    data.extend(exact_matches)
    
    # 2. 与A的高相似度文本对应（相似度0.9+，5条）
    high_similarity_bases = [
        "federated learning enables collaborative model training",
        "secure multi-party computation protects data privacy",
        "gradient descent optimization minimizes loss functions",
        "convolutional neural networks excel at image recognition",
        "natural language processing understands human communication"
    ]
    for base in high_similarity_bases:
        # 生成与A不同的变体，但同样相似
        data.append(generate_similar_text(base, 0.93))
    
    # 3. 与A的中等相似度文本对应（相似度0.8+，5条）
    medium_similarity_bases = [
        "distributed systems handle large scale data processing",
        "encryption algorithms secure sensitive information",
        "deep learning models require substantial computational resources",
        "data preprocessing improves model performance",
        "feature extraction identifies relevant patterns"
    ]
    for base in medium_similarity_bases:
        data.append(generate_similar_text(base, 0.82))
    
    # 4. 与A的低相似度文本对应（相似度0.7+，5条）
    low_similarity_bases = [
        "blockchain technology ensures data integrity",
        "cloud computing provides scalable infrastructure",
        "edge computing reduces network latency",
        "internet of things connects physical devices",
        "big data analytics extracts valuable insights"
    ]
    for base in low_similarity_bases:
        data.append(generate_similar_text(base, 0.78))
    
    # 5. 长文本（3条，与A的长文本相似但不完全相同）
    long_texts = [
        "Federated learning represents a significant advancement in machine learning, allowing various organizations to jointly build a shared model while keeping their data localized. This approach effectively solves privacy issues and makes use of diverse data sources.",
        "Organizations worldwide must now navigate complex data privacy laws like GDPR and CCPA. Meeting these standards demands strong technical protections and clear policies for managing data throughout the company.",
        "Recent progress in deep learning, particularly with transformer models and attention-based architectures, has dramatically improved how machines understand language. Today's models can translate languages, create summaries, and answer questions with impressive precision."
    ]
    data.extend(long_texts)
    
    # 6. 包含特殊字符的文本（5条，部分与A相似）
    special_format_texts = [
        "Error code: 0x404 - File not found at /path/to/resource",
        "Timestamp: 2024-01-15 14:30:00 | User ID: 12345 | Action: LOGIN_SUCCESS",
        "function calculate(a, b) { return a * b + Math.sqrt(a); }",
        "SELECT id, name FROM users WHERE age >= 18 AND status = 'active';",
        "Response: {\"status\": \"success\", \"count\": 5, \"data\": [1, 2, 3, 4, 5]}"
    ]
    data.extend(special_format_texts)
    
    # 7. 完全不同文本（5条，与A的不同）
    different_texts = [
        generate_random_string(52),
        generate_random_string(58),
        generate_random_string(48),
        generate_random_string(53),
        generate_random_string(42)
    ]
    data.extend(different_texts)
    
    return data

def main():
    # 生成参与方A的数据
    party_a_data = generate_party_a_data()
    with open('test/party_a_complex.txt', 'w', encoding='utf-8') as f:
        for item in party_a_data:
            f.write(item + '\n')
    
    # 生成参与方B的数据
    party_b_data = generate_party_b_data()
    with open('test/party_b_complex.txt', 'w', encoding='utf-8') as f:
        for item in party_b_data:
            f.write(item + '\n')
    
    print("复杂测试数据生成成功！")
    print(f"参与方A数据: {len(party_a_data)} 条记录 -> test/party_a_complex.txt")
    print(f"参与方B数据: {len(party_b_data)} 条记录 -> test/party_b_complex.txt")
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
