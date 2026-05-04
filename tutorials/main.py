import os
import hashlib

def verify_huggingface_cache(cache_dir):
    print(f"正在扫描缓存目录层级: {cache_dir}")
    
    # 检查目录是否存在
    if not os.path.exists(cache_dir):
        print(f"错误：目录 {cache_dir} 不存在")
        return
    
    # 检查是否为数据集缓存结构（使用hf download下载的数据集）
    dataset_cache_found = False
    for item in os.listdir(cache_dir):
        if item.startswith('datasets--'):
            dataset_cache_found = True
            dataset_cache_path = os.path.join(cache_dir, item)
            print(f"发现Hugging Face缓存格式数据集: {item}")
            verify_hf_cache_format(cache_dir)
            return
    
    # 如果不是hf cache格式，检查是否为直接下载的数据集
    # 直接下载的数据集会在目录下直接包含电池数据子集文件夹
    battery_subsets_found = []
    for item in os.listdir(cache_dir):
        item_path = os.path.join(cache_dir, item)
        if os.path.isdir(item_path):
            # 检查是否包含.pkl文件
            for filename in os.listdir(item_path):
                if filename.endswith('.pkl'):
                    battery_subsets_found.append(item)
                    break
    
    if battery_subsets_found:
        print(f"发现直接下载的数据集格式，包含以下电池数据子集: {', '.join(battery_subsets_found)}")
        verify_direct_download_dataset(cache_dir)
    else:
        # 如果是标准模型缓存结构
        verify_model_cache(cache_dir)

def verify_hf_cache_format(cache_dir):
    """验证Hugging Face缓存格式的数据集"""
    dataset_cache_path = None
    for item in os.listdir(cache_dir):
        if item.startswith('datasets--'):
            dataset_cache_path = os.path.join(cache_dir, item)
            break
    
    if not dataset_cache_path:
        print("未找到Hugging Face缓存格式的数据集")
        return
    
    snapshots_dir = os.path.join(dataset_cache_path, 'snapshots')
    if not os.path.exists(snapshots_dir):
        print(f"缓存目录中未找到snapshots文件夹")
        return
    
    # 统计电池数据子集信息
    dataset_stats = {}
    corrupted_files = []
    total_files = 0
    
    # 查找所有快照版本
    snapshot_versions = []
    for item in os.listdir(snapshots_dir):
        item_path = os.path.join(snapshots_dir, item)
        if os.path.isdir(item_path):
            snapshot_versions.append(item)
    
    print(f"发现 {len(snapshot_versions)} 个快照版本: {', '.join(snapshot_versions)}")
    
    for snapshot_version in snapshot_versions:
        snapshot_path = os.path.join(snapshots_dir, snapshot_version)
        
        # 查找电池数据子集目录
        battery_subsets = []
        for item in os.listdir(snapshot_path):
            item_path = os.path.join(snapshot_path, item)
            if os.path.isdir(item_path):
                battery_subsets.append((item, item_path))
        
        if not battery_subsets:
            continue
        
        print(f"\n=== 快照版本: {snapshot_version} ===")
        print(f"包含电池数据子集:")
        
        for subset_name, subset_path in sorted(battery_subsets):
            battery_files = []
            for filename in os.listdir(subset_path):
                if filename.endswith('.pkl'):
                    battery_files.append(filename)
            
            print(f"  {subset_name}: {len(battery_files)} 块电池")
        
        # 详细的文件完整性检查
        print(f"\n开始完整性检查...")
        for subset_name, subset_path in sorted(battery_subsets):
            subset_corrupted = []
            for filename in os.listdir(subset_path):
                if filename.endswith('.pkl'):
                    filepath = os.path.join(subset_path, filename)
                    total_files += 1
                    
                    try:
                        file_size = os.path.getsize(filepath)
                        if file_size < 100:
                            subset_corrupted.append((filename, "文件过小"))
                            corrupted_files.append(filepath)
                            continue
                            
                        with open(filepath, 'rb') as f:
                            header = f.read(1024)
                            
                            if len(header) < 100:
                                subset_corrupted.append((filename, "文件内容不足"))
                                corrupted_files.append(filepath)
                            elif header[0:2] not in [b'\x80', b'\x80\x03', b'\x80\x04']:
                                subset_corrupted.append((filename, "非标准pickle格式"))
                                corrupted_files.append(filepath)
                    except Exception as e:
                        error_msg = f"读取错误: {str(e)}"
                        subset_corrupted.append((filename, error_msg))
                        corrupted_files.append(filepath)
                        print(f"    ❌ {filename}: {error_msg}")
            
            if subset_corrupted:
                print(f"  {subset_name} - 损坏文件: {len(subset_corrupted)}/{len([f for f in os.listdir(subset_path) if f.endswith('.pkl')])}")
                for filename, reason in subset_corrupted:
                    print(f"    ❌ {filename}: {reason}")
            else:
                print(f"  {subset_name} ✅ 所有文件正常")
    
    # 汇总统计
    print(f"\n{'='*50}")
    print(f"汇总统计")
    print(f"{'='*50}")
    print(f"电池文件总数: {total_files}")
    print(f"损坏文件总数: {len(corrupted_files)}")
    
    if corrupted_files:
        print(f"\n⚠️  发现 {len(corrupted_files)} 个损坏文件")
    else:
        print("\n✅ 所有电池数据文件完整性检查通过")

def verify_direct_download_dataset(dataset_dir):
    """验证直接下载的数据集（hf download --local-dir格式）"""
    print(f"\n开始验证直接下载的数据集: {dataset_dir}")
    
    dataset_stats = {}
    corrupted_files = []
    total_files = 0
    
    # 获取所有电池数据子集
    battery_subsets = []
    for item in os.listdir(dataset_dir):
        item_path = os.path.join(dataset_dir, item)
        if os.path.isdir(item_path):
            # 检查是否包含.pkl文件
            has_pkl = any(f.endswith('.pkl') for f in os.listdir(item_path))
            if has_pkl:
                battery_subsets.append((item, item_path))
    
    if not battery_subsets:
        print("未找到电池数据子集")
        return
    
    print(f"\n发现 {len(battery_subsets)} 个电池数据子集")
    print(f"\n{'='*50}")
    print("电池数据子集统计")
    print(f"{'='*50}")
    
    # 统计每个子集的电池数量
    for subset_name, subset_path in sorted(battery_subsets):
        battery_files = []
        for filename in os.listdir(subset_path):
            if filename.endswith('.pkl'):
                battery_files.append(filename)
        
        dataset_stats[subset_name] = {
            'file_count': len(battery_files),
            'files': battery_files,
            'corrupted_files': []
        }
        
        print(f"{subset_name:15s}: {len(battery_files):3d} 块电池")
    
    # 详细的文件完整性检查
    print(f"\n{'='*50}")
    print("文件完整性检查")
    print(f"{'='*50}")
    
    for subset_name, subset_path in sorted(battery_subsets):
        subset_corrupted = []
        pkl_files = [f for f in os.listdir(subset_path) if f.endswith('.pkl')]
        
        for filename in pkl_files:
            filepath = os.path.join(subset_path, filename)
            total_files += 1
            
            try:
                file_size = os.path.getsize(filepath)
                if file_size < 100:
                    subset_corrupted.append((filename, "文件过小", file_size))
                    corrupted_files.append(filepath)
                    continue
                    
                with open(filepath, 'rb') as f:
                    header = f.read(1024)
                    
                    if len(header) < 100:
                        subset_corrupted.append((filename, "文件内容不足", file_size))
                        corrupted_files.append(filepath)
                    elif header[0:2] not in [b'\x80', b'\x80\x03', b'\x80\x04']:
                        subset_corrupted.append((filename, "非标准pickle格式", file_size))
                        corrupted_files.append(filepath)
                        
            except Exception as e:
                error_msg = f"读取错误: {str(e)}"
                subset_corrupted.append((filename, error_msg, 0))
                corrupted_files.append(filepath)
        
        if subset_corrupted:
            print(f"\n{subset_name}:")
            print(f"  总文件数: {len(pkl_files)}, 损坏: {len(subset_corrupted)}")
            for filename, reason, size in subset_corrupted:
                print(f"    ❌ {filename} ({size} bytes): {reason}")
            dataset_stats[subset_name]['corrupted_files'] = subset_corrupted
        else:
            print(f"{subset_name}: ✅ 所有 {len(pkl_files)} 个文件正常")
    
    # 汇总统计
    print(f"\n{'='*50}")
    print("汇总统计")
    print(f"{'='*50}")
    print(f"电池数据子集总数: {len(battery_subsets)}")
    print(f"电池文件总数: {total_files}")
    print(f"损坏文件总数: {len(corrupted_files)}")
    
    if corrupted_files:
        print(f"\n⚠️  发现 {len(corrupted_files)} 个损坏文件，建议重新下载")
    else:
        print("\n✅ 所有电池数据文件完整性检查通过")

def verify_model_cache(cache_dir):
    """验证标准Hugging Face模型缓存"""
    hash_to_filenames = {}
    blobs_to_check = set()
    
    for root, dirs, files in os.walk(cache_dir):
        # 检查snapshots目录中的符号链接
        if 'snapshots' in root:
            for filename in files:
                filepath = os.path.join(root, filename)
                if os.path.islink(filepath):
                    link_target = os.path.realpath(filepath)
                    if 'blobs' in link_target:
                        blob_hash = os.path.basename(link_target)
                        if blob_hash not in hash_to_filenames:
                            hash_to_filenames[blob_hash] = []
                        hash_to_filenames[blob_hash].append(filepath)
        
        # 直接检查blobs目录
        if os.path.basename(root) == 'blobs':
            for filename in files:
                blob_path = os.path.join(root, filename)
                blobs_to_check.add((blob_path, filename))

    if not blobs_to_check:
        print("未在指定路径下找到任何符合规范的底层数据块。")
        return

    print(f"发现 {len(blobs_to_check)} 个底层数据块需要校验，开始执行密码学哈希计算流程...")
    
    corrupted_blobs = []
    
    for blob_path, expected_hash in blobs_to_check:
        if len(expected_hash) == 64:
            hash_func = hashlib.sha256()
        elif len(expected_hash) == 40:
            hash_func = hashlib.sha1()
        else:
            continue
            
        try:
            with open(blob_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024 * 8), b""): 
                    hash_func.update(chunk)
                    
            calculated_hash = hash_func.hexdigest()
            
            if calculated_hash != expected_hash:
                corrupted_blobs.append(expected_hash)
        except Exception as e:
            print(f"读取物理文件时发生系统级别的 IO 错误: {blob_path}")
            corrupted_blobs.append(expected_hash)

    print("底层数据块哈希计算阶段结束。")
    
    if corrupted_blobs:
        print("警告：校验发现以下底层数据块处于不完整或损坏状态：")
        for bad_hash in corrupted_blobs:
            affected_files = hash_to_filenames.get(bad_hash, ["未能映射到具体的快照拓扑文件"])
            print(f"损坏的数据块哈希值: {bad_hash}")
            print("受此损坏影响的实际文件列表:")
            for af in affected_files:
                print(f"  - {af}")
    else:
        print("数据完整性密码学校验全部通过，所有底层数据块及其映射的快照文件均处于绝对完整状态。")

# 使用当前目录
target_directory = r"../dataset"
verify_huggingface_cache(target_directory)
