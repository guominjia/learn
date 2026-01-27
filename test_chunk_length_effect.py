"""
测试chunk长度对相似度的影响
"""
import chromadb
from chromadb.config import Settings

def test_chunk_length_vs_similarity():
    """演示长短chunk对相似度的影响"""
    
    # 初始化ChromaDB
    client = chromadb.Client(Settings(
        anonymized_telemetry=False,
        is_persistent=False
    ))
    collection = client.create_collection("test_length")
    
    # 准备测试数据
    query = "如何初始化DDR4内存控制器？"
    
    # Chunk 1: 短而聚焦（100字）
    short_chunk = """
    DDR4内存控制器初始化步骤：
    1. 读取SPD信息获取内存参数
    2. 配置时序寄存器（CAS延迟、RAS to CAS延迟）
    3. 执行DRAM训练流程
    4. 验证内存稳定性
    """
    
    # Chunk 2: 中等长度，包含相关+无关内容（300字）
    medium_chunk = """
    第5章 内存子系统管理
    
    5.1 DDR4内存控制器初始化
    初始化步骤包括读取SPD、配置时序、执行训练和验证稳定性。
    
    5.2 内存错误检测与纠正
    ECC功能可以检测和纠正单比特错误，检测双比特错误。
    
    5.3 内存性能优化
    通过调整预取策略、刷新间隔和功耗模式来优化性能。
    
    5.4 内存兼容性测试
    确保不同厂商的内存条都能正常工作。
    """
    
    # Chunk 3: 长而分散，多个主题（600字）
    long_chunk = """
    第3章 BIOS固件架构详解
    
    3.1 初始化流程概述
    BIOS启动时需要初始化所有硬件组件，包括CPU、内存、PCI设备、
    存储控制器、网络接口等。整个流程遵循严格的顺序。
    
    3.2 CPU初始化
    配置微码、缓存、电源管理、频率等。检测CPU功能并启用特性。
    
    3.3 内存初始化
    DDR4内存控制器初始化包括SPD读取、时序配置和训练流程。
    需要处理不同容量、频率和时序的内存条。
    
    3.4 PCI设备枚举
    扫描PCI总线，分配资源，初始化设备驱动。支持PCIe Gen3/Gen4。
    
    3.5 存储设备检测
    检测SATA和NVMe设备，读取分区表，准备启动。
    
    3.6 网络启动支持
    实现PXE协议，支持从网络加载操作系统。
    
    3.7 安全启动
    验证固件和操作系统的数字签名，防止恶意代码。
    
    3.8 电源管理
    配置ACPI表，实现睡眠、休眠等节能功能。
    """
    
    # 添加到collection
    collection.add(
        documents=[short_chunk, medium_chunk, long_chunk],
        ids=["short", "medium", "long"],
        metadatas=[
            {"length": len(short_chunk), "topics": 1},
            {"length": len(medium_chunk), "topics": 4},
            {"length": len(long_chunk), "topics": 8}
        ]
    )
    
    # 执行查询
    results = collection.query(
        query_texts=query,
        n_results=3
    )
    
    # 分析结果
    print("\n" + "="*80)
    print(f"查询: {query}")
    print("="*80)
    
    for i, (doc_id, distance, metadata) in enumerate(zip(
        results['ids'][0],
        results['distances'][0],
        results['metadatas'][0]
    ), 1):
        similarity = 1 - (distance / 2)  # 转换为相似度
        
        print(f"\n排名 #{i}: {doc_id.upper()}")
        print(f"  相似度: {similarity:.4f}")
        print(f"  距离: {distance:.4f}")
        print(f"  长度: {metadata['length']} 字符")
        print(f"  主题数: {metadata['topics']}")
        print(f"  字符/主题: {metadata['length'] / metadata['topics']:.0f}")
    
    print("\n" + "="*80)
    print("观察:")
    print("="*80)
    
    # 检查是否出现"长度惩罚"效应
    similarities = [1 - (d/2) for d in results['distances'][0]]
    lengths = [m['length'] for m in results['metadatas'][0]]
    topics = [m['topics'] for m in results['metadatas'][0]]
    
    # 计算相关性
    import numpy as np
    if len(similarities) > 1:
        corr_length = np.corrcoef(lengths, similarities)[0, 1]
        corr_topics = np.corrcoef(topics, similarities)[0, 1]
        
        print(f"相似度 vs 长度的相关性: {corr_length:.3f}")
        print(f"相似度 vs 主题数的相关性: {corr_topics:.3f}")
        
        if corr_length < -0.5:
            print("\n⚠️  警告: 检测到强负相关 - 长文档被惩罚！")
            print("   建议: 减小chunk_size或使用句子级检索")
        
        if corr_topics < -0.5:
            print("\n⚠️  警告: 多主题导致相似度降低！")
            print("   建议: 优化分块策略，确保每个chunk聚焦单一主题")
    
    print("\n预期:")
    print("  理想情况: short > medium > long (因为聚焦度依次降低)")
    print(f"  实际排名: {results['ids'][0]}")
    
    # 清理
    client.delete_collection("test_length")
    
    return results

if __name__ == "__main__":
    test_chunk_length_vs_similarity()
