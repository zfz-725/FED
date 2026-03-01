#!/usr/bin/env python3
"""
验证PSI测试结果
分析匹配的准确性和相似度
"""

import struct
import sys

def read_signatures(filename):
    """读取签名文件"""
    signatures = []
    with open(filename, 'rb') as f:
        while True:
            len_bytes = f.read(4)
            if not len_bytes:
                break
            text_len = struct.unpack('i', len_bytes)[0]
            text = f.read(text_len).decode('utf-8')
            sig_bytes = f.read(128 * 4)
            signatures.append((text, sig_bytes))
    return signatures

def read_candidates(filename):
    """读取候选对"""
    candidates = []
    with open(filename, 'rb') as f:
        while True:
            idx_bytes = f.read(8)
            if not idx_bytes or len(idx_bytes) < 8:
                break
            a_idx, b_idx = struct.unpack('ii', idx_bytes)
            candidates.append((a_idx, b_idx))
    return candidates

def compute_jaccard_similarity(text1, text2, shingle_len=5):
    """计算Jaccard相似度"""
    def get_shingles(text, k):
        return set(text[i:i+k] for i in range(len(text) - k + 1))
    
    shingles1 = get_shingles(text1, shingle_len)
    shingles2 = get_shingles(text2, shingle_len)
    
    intersection = len(shingles1 & shingles2)
    union = len(shingles1 | shingles2)
    
    return intersection / union if union > 0 else 0.0

def main():
    if len(sys.argv) < 3:
        print("用法: python3 verify_psi_results.py <party_a_file> <party_b_file> [candidate_file]")
        return
    
    party_a_file = sys.argv[1]
    party_b_file = sys.argv[2]
    candidate_file = sys.argv[3] if len(sys.argv) > 3 else "test/output_complex/candidate_indices.bin"
    
    print("=" * 80)
    print("PSI测试结果验证")
    print("=" * 80)
    
    # 读取数据
    print("\n1. 读取参与方A的数据...")
    a_data = read_signatures(party_a_file)
    print(f"   读取了 {len(a_data)} 条记录")
    
    print("\n2. 读取参与方B的数据...")
    b_data = read_signatures(party_b_file)
    print(f"   读取了 {len(b_data)} 条记录")
    
    print("\n3. 读取候选对...")
    candidates = read_candidates(candidate_file)
    print(f"   找到 {len(candidates)} 个候选对")
    
    # 分析匹配结果
    print("\n" + "=" * 80)
    print("匹配结果分析")
    print("=" * 80)
    
    exact_matches = 0
    high_similarity = 0
    medium_similarity = 0
    low_similarity = 0
    
    for i, (a_idx, b_idx) in enumerate(candidates):
        if a_idx < len(a_data) and b_idx < len(b_data):
            text_a = a_data[a_idx][0]
            text_b = b_data[b_idx][0]
            similarity = compute_jaccard_similarity(text_a, text_b)
            
            print(f"\n候选对 {i+1}:")
            print(f"  参与方A索引: {a_idx}")
            print(f"  参与方B索引: {b_idx}")
            print(f"  相似度: {similarity:.4f}")
            print(f"  参与方A文本: {text_a[:100]}{'...' if len(text_a) > 100 else ''}")
            print(f"  参与方B文本: {text_b[:100]}{'...' if len(text_b) > 100 else ''}")
            
            if similarity == 1.0:
                exact_matches += 1
                print("  [精确匹配]")
            elif similarity >= 0.8:
                high_similarity += 1
                print("  [高相似度]")
            elif similarity >= 0.6:
                medium_similarity += 1
                print("  [中等相似度]")
            else:
                low_similarity += 1
                print("  [低相似度]")
    
    # 统计信息
    print("\n" + "=" * 80)
    print("统计摘要")
    print("=" * 80)
    print(f"总候选对数: {len(candidates)}")
    print(f"精确匹配 (相似度=1.0): {exact_matches}")
    print(f"高相似度 (0.8-1.0): {high_similarity}")
    print(f"中等相似度 (0.6-0.8): {medium_similarity}")
    print(f"低相似度 (<0.6): {low_similarity}")
    
    # 计算准确率（假设相似度>=0.7为正确匹配）
    correct_matches = exact_matches + high_similarity + medium_similarity
    if len(candidates) > 0:
        accuracy = correct_matches / len(candidates)
        print(f"\n准确率 (相似度>=0.6): {accuracy:.2%}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
