"""
对比不同池化策略的效果
演示为什么Embedding模型需要平均而不是取最后一个token
"""
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

def compare_pooling_strategies():
    """对比CLS、Last Token、Mean Pooling的区别"""
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # 测试文本
    texts = [
        "内存初始化",  # 短文本
        "内存初始化和PCI设备枚举",  # 中等
        "内存初始化是BIOS的重要步骤，包括SPD读取和时序配置。此外还有PCI设备枚举、硬盘检测、网络启动等功能。"  # 长文本
    ]
    
    query = "内存初始化"
    
    print("="*80)
    print("对比不同池化策略对相似度的影响")
    print("="*80)
    
    # 获取query的embedding（使用mean pooling）
    query_inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        query_outputs = model(**query_inputs)
        query_embedding = mean_pooling(query_outputs.last_hidden_state, query_inputs['attention_mask'])
    
    for text in texts:
        print(f"\n文本: {text[:50]}...")
        print("-"*80)
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        # 获取hidden states
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state  # [1, seq_len, 768]
        
        seq_len = last_hidden_state.shape[1]
        print(f"Token数量: {seq_len}")
        
        # 策略1: CLS Token (第一个token)
        cls_embedding = last_hidden_state[:, 0, :]  # [1, 768]
        
        # 策略2: Last Token (最后一个token)
        last_token_embedding = last_hidden_state[:, -1, :]  # [1, 768]
        
        # 策略3: Mean Pooling (平均所有token)
        mean_embedding = mean_pooling(last_hidden_state, inputs['attention_mask'])
        
        # 策略4: Max Pooling
        max_embedding = torch.max(last_hidden_state, dim=1)[0]
        
        # 策略5: First & Last Average
        first_last_embedding = (last_hidden_state[:, 0, :] + last_hidden_state[:, -1, :]) / 2
        
        # 计算与query的相似度
        similarities = {
            "CLS Token": cosine_similarity(query_embedding, cls_embedding),
            "Last Token": cosine_similarity(query_embedding, last_token_embedding),
            "Mean Pooling": cosine_similarity(query_embedding, mean_embedding),
            "Max Pooling": cosine_similarity(query_embedding, max_embedding),
            "First+Last Avg": cosine_similarity(query_embedding, first_last_embedding),
        }
        
        # 打印结果
        for strategy, sim in similarities.items():
            print(f"  {strategy:15s}: {sim:.4f}")
        
        # 分析token级别的重要性
        print(f"\n  Token分析:")
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        token_norms = torch.norm(last_hidden_state[0], dim=1).numpy()
        
        # 找出范数最大的3个token（最"重要"的）
        top_indices = np.argsort(token_norms)[-3:][::-1]
        print(f"    最重要的3个token:")
        for idx in top_indices:
            if idx < len(tokens):
                print(f"      位置{idx}: '{tokens[idx]}' (范数: {token_norms[idx]:.3f})")
        
        # 检查位置偏见
        first_half_norm = token_norms[:seq_len//2].mean()
        second_half_norm = token_norms[seq_len//2:].mean()
        print(f"    前半部分平均范数: {first_half_norm:.3f}")
        print(f"    后半部分平均范数: {second_half_norm:.3f}")
        
        if abs(first_half_norm - second_half_norm) > 0.5:
            print(f"    ⚠️  检测到位置偏见！")

def mean_pooling(last_hidden_state, attention_mask):
    """Mean Pooling - 考虑attention mask"""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def cosine_similarity(a, b):
    """计算余弦相似度"""
    a = a.squeeze()
    b = b.squeeze()
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))

def demonstrate_decoder_vs_encoder():
    """演示Decoder和Encoder的区别"""
    print("\n" + "="*80)
    print("Decoder (GPT) vs Encoder (BERT) 的Attention模式")
    print("="*80)
    
    seq_len = 5
    print(f"\n假设输入序列长度: {seq_len}")
    print("\nDecoder (GPT) - 因果注意力矩阵:")
    print("     t1  t2  t3  t4  t5  (输入)")
    decoder_attention = np.tril(np.ones((seq_len, seq_len)))
    for i, row in enumerate(decoder_attention):
        print(f"  t{i+1} {' '.join(['✓' if x else '✗' for x in row])}  ← t{i+1}只能看到前{i+1}个token")
    
    print("\n结果: t5看到所有token → h5包含全局信息 → 可以单独使用")
    
    print("\nEncoder (BERT) - 双向注意力矩阵:")
    print("     t1  t2  t3  t4  t5  (输入)")
    encoder_attention = np.ones((seq_len, seq_len))
    for i, row in enumerate(encoder_attention):
        print(f"  t{i+1} {' '.join(['✓' for _ in row])}  ← t{i+1}看到所有token，但从自己的角度")
    
    print("\n结果: 每个token都看到全局，但角度不同 → 需要池化 → 平均所有token")
    
    print("\n" + "="*80)
    print("关键差异:")
    print("="*80)
    print("1. GPT (Decoder):")
    print("   - 因果注意力 → 信息逐步积累到最后")
    print("   - h_last = f(h1, h2, ..., h_{last-1})")
    print("   - 最后一个token天然是'总结'")
    print("")
    print("2. BERT (Encoder):")
    print("   - 双向注意力 → 每个token都是'平等'的")
    print("   - h_i = f(h_all), 但从位置i的视角")
    print("   - 需要平均消除位置偏见")

if __name__ == "__main__":
    print("测试需要下载模型，可能需要一些时间...\n")
    try:
        demonstrate_decoder_vs_encoder()
        print("\n开始实际测试...")
        compare_pooling_strategies()
    except Exception as e:
        print(f"错误: {e}")
        print("\n如果是模型下载问题，可以先运行 demonstrate_decoder_vs_encoder() 查看理论分析")
