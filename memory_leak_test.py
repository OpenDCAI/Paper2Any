"""
memory_leak_test.py
独立的内存泄漏检测脚本 - 放在项目根目录运行
"""

import asyncio
import gc
import json
import os
import sys
import tempfile
import tracemalloc
from pathlib import Path
from typing import List, Dict, Any, Set
import psutil
import re
import pandas as pd
import uuid
import textwrap


# ============================================================================
#                          复制需要测试的函数
# ============================================================================

def _patch_first_entry_file(py_file: str | Path,
                            old_path: str,
                            new_path: str) -> None:
    """从原代码复制"""
    py_file = Path(py_file).expanduser().resolve()
    code = py_file.read_text(encoding="utf-8")

    pattern = (
        r'first_entry_file_name\s*=\s*[\'"]'
        + re.escape(old_path)
        + r'[\'"]'
    )
    replacement = f'first_entry_file_name=\"{new_path}\"'
    new_code, n = re.subn(pattern, replacement, code, count=1)
    if n == 0:
        new_code = code.replace(old_path, new_path)

    py_file.write_text(new_code, encoding="utf-8")


def _ensure_py_file(code: str, file_name: str | None = None) -> Path:
    """从原代码复制"""
    if file_name:
        target = Path(file_name).expanduser().resolve()
    else:
        target = Path(tempfile.gettempdir()) / f"recommend_pipeline_{uuid.uuid4().hex}.py"
    target.write_text(textwrap.dedent(code), encoding="utf-8")
    print(f"[Test] pipeline code written to {target}")
    return target


def _create_debug_sample(src_file: str | Path, sample_lines: int = 10) -> Path:
    """从原代码复制"""
    src_path = Path(src_file).expanduser().resolve()
    if not src_path.is_file():
        raise FileNotFoundError(f"source file not found: {src_path}")

    tmp_path = (
        Path(tempfile.gettempdir())
        / f"{src_path.stem}_sample_{sample_lines}{src_path.suffix}"
    )

    suffix = src_path.suffix.lower()
    
    if suffix == '.csv':
        df = pd.read_csv(src_path)
        sample_df = df.head(sample_lines)
        sample_df.to_csv(tmp_path, index=False, encoding="utf-8")
        print(f"[Test] CSV sample written to {tmp_path}")
    
    elif suffix == '.json':
        with src_path.open("r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            
            if first_char == '[':
                data = json.load(f)
                sample_data = data[:sample_lines]
                with tmp_path.open("w", encoding="utf-8") as wf:
                    json.dump(sample_data, wf, ensure_ascii=False, indent=2)
            else:
                sample_data = []
                for idx, line in enumerate(f):
                    if idx >= sample_lines:
                        break
                    sample_data.append(json.loads(line.strip()))
                
                with tmp_path.open("w", encoding="utf-8") as wf:
                    for item in sample_data:
                        wf.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        print(f"[Test] JSON sample written to {tmp_path}")
    
    elif suffix == '.jsonl':
        sample_data = []
        with src_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= sample_lines:
                    break
                sample_data.append(json.loads(line.strip()))
        
        with tmp_path.open("w", encoding="utf-8") as wf:
            for item in sample_data:
                wf.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        print(f"[Test] JSONL sample written to {tmp_path}")
    
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    return tmp_path


async def _run_py(file_path: Path) -> dict:
    """从原代码复制"""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(file_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, stderr_b = await proc.communicate()
    stdout, stderr = stdout_b.decode(), stderr_b.decode()

    return {
        "success": proc.returncode == 0,
        "return_code": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "file_path": str(file_path),
    }


# ============================================================================
#                          内存泄漏检测器
# ============================================================================

class MemoryLeakDetector:
    """内存泄漏检测器"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.snapshots = []
        self.temp_files_created = []
        
    def start_tracking(self):
        """开始追踪内存"""
        gc.collect()
        tracemalloc.start()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024
        print(f"[MemoryCheck] 基线内存: {self.baseline_memory:.2f} MB")
        
    def take_snapshot(self, label: str):
        """拍摄内存快照"""
        gc.collect()
        current_memory = self.process.memory_info().rss / 1024 / 1024
        snapshot = tracemalloc.take_snapshot()
        
        self.snapshots.append({
            'label': label,
            'memory_mb': current_memory,
            'memory_delta': current_memory - self.baseline_memory,
            'snapshot': snapshot
        })
        
        print(f"[MemoryCheck] {label}: {current_memory:.2f} MB "
              f"(+{current_memory - self.baseline_memory:.2f} MB)")
        
    def track_temp_file(self, file_path: Path):
        """记录创建的临时文件"""
        self.temp_files_created.append(file_path)
        
    def check_temp_file_cleanup(self) -> Dict[str, Any]:
        """检查临时文件是否被清理"""
        uncleaned = []
        total_size = 0
        
        for fp in self.temp_files_created:
            if fp.exists():
                size = fp.stat().st_size
                uncleaned.append({
                    'path': str(fp),
                    'size_kb': size / 1024
                })
                total_size += size
                
        return {
            'uncleaned_count': len(uncleaned),
            'total_size_mb': total_size / 1024 / 1024,
            'files': uncleaned
        }
        
    def compare_snapshots(self, index1: int, index2: int, top_n: int = 10):
        """比较两个快照"""
        if index1 >= len(self.snapshots) or index2 >= len(self.snapshots):
            return
            
        snap1 = self.snapshots[index1]['snapshot']
        snap2 = self.snapshots[index2]['snapshot']
        
        stats = snap2.compare_to(snap1, 'lineno')
        
        print(f"\n[MemoryCheck] Top {top_n} 内存增长:")
        print(f"从 '{self.snapshots[index1]['label']}' 到 '{self.snapshots[index2]['label']}'")
        print("-" * 80)
        
        for stat in stats[:top_n]:
            print(f"{stat}")
            
    def stop_tracking(self):
        """停止追踪"""
        tracemalloc.stop()
        
    def generate_report(self) -> str:
        """生成检测报告"""
        report = ["=" * 80]
        report.append("内存泄漏检测报告")
        report.append("=" * 80)
        
        report.append("\n1. 内存增长趋势:")
        for snap in self.snapshots:
            report.append(f"  {snap['label']}: {snap['memory_mb']:.2f} MB "
                         f"(+{snap['memory_delta']:.2f} MB)")
            
        cleanup_status = self.check_temp_file_cleanup()
        report.append(f"\n2. 临时文件清理:")
        report.append(f"  创建文件总数: {len(self.temp_files_created)}")
        report.append(f"  未清理文件数: {cleanup_status['uncleaned_count']}")
        report.append(f"  占用空间: {cleanup_status['total_size_mb']:.2f} MB")
        
        if cleanup_status['files']:
            report.append("  未清理文件列表 (前10个):")
            for f in cleanup_status['files'][:10]:
                report.append(f"    - {f['path']} ({f['size_kb']:.2f} KB)")
                
        report.append("\n3. 泄漏风险评估:")
        total_growth = self.snapshots[-1]['memory_delta'] if self.snapshots else 0
        
        risk_items = []
        if total_growth > 100:
            risk_items.append("⚠️  内存增长超过100MB")
        elif total_growth > 50:
            risk_items.append("⚠️  内存增长超过50MB")
            
        if cleanup_status['uncleaned_count'] > 10:
            risk_items.append(f"⚠️  有 {cleanup_status['uncleaned_count']} 个临时文件未清理")
        elif cleanup_status['uncleaned_count'] > 0:
            risk_items.append(f"⚠️  有 {cleanup_status['uncleaned_count']} 个临时文件未清理")
            
        if not risk_items:
            report.append("  ✓ 低风险: 内存增长在可接受范围，临时文件管理良好")
        else:
            for item in risk_items:
                report.append(f"  {item}")
            
        report.append("=" * 80)
        return "\n".join(report)


# ============================================================================
#                              测试用例
# ============================================================================

async def test_temp_file_creation():
    """测试1: 临时文件创建"""
    print("\n" + "=" * 80)
    print("测试1: 临时文件创建与清理")
    print("=" * 80)
    
    detector = MemoryLeakDetector()
    detector.start_tracking()
    detector.take_snapshot("初始状态")
    
    print("\n创建 20 个临时 Python 文件...")
    for i in range(20):
        code = f"print('test {i}')\nimport time\ntime.sleep(0.01)"
        py_file = _ensure_py_file(code)
        detector.track_temp_file(py_file)
        
    detector.take_snapshot("创建20个文件后")
    
    # 检查文件清理
    cleanup_status = detector.check_temp_file_cleanup()
    print(f"\n临时文件状态:")
    print(f"  创建: {len(detector.temp_files_created)} 个")
    print(f"  未清理: {cleanup_status['uncleaned_count']} 个")
    print(f"  占用: {cleanup_status['total_size_mb']:.2f} MB")
    
    print("\n" + detector.generate_report())
    detector.stop_tracking()
    
    return cleanup_status


async def test_debug_sample_creation():
    """测试2: 调试样本创建"""
    print("\n" + "=" * 80)
    print("测试2: 调试样本文件创建")
    print("=" * 80)
    
    detector = MemoryLeakDetector()
    detector.start_tracking()
    detector.take_snapshot("初始状态")
    
    # 创建测试数据文件
    test_files = {}
    
    # CSV
    csv_file = Path(tempfile.gettempdir()) / "test_data.csv"
    csv_file.write_text("col1,col2,col3\n" + "\n".join(
        [f"{i},{i*2},{i*3}" for i in range(1000)]
    ))
    detector.track_temp_file(csv_file)
    test_files['csv'] = csv_file
    
    # JSONL
    jsonl_file = Path(tempfile.gettempdir()) / "test_data.jsonl"
    with jsonl_file.open('w') as f:
        for i in range(1000):
            f.write(json.dumps({'id': i, 'value': i*2}) + '\n')
    detector.track_temp_file(jsonl_file)
    test_files['jsonl'] = jsonl_file
    
    # JSON Array
    json_file = Path(tempfile.gettempdir()) / "test_data.json"
    json_file.write_text(json.dumps([{'id': i, 'value': i*2} for i in range(1000)]))
    detector.track_temp_file(json_file)
    test_files['json'] = json_file
    
    detector.take_snapshot("创建测试文件后")
    
    # 创建样本
    print("\n创建样本文件...")
    for fmt, file_path in test_files.items():
        for i in range(3):
            sample_file = _create_debug_sample(file_path, sample_lines=10)
            detector.track_temp_file(sample_file)
            print(f"  {fmt} 样本 {i+1}: {sample_file.stat().st_size} bytes")
    
    detector.take_snapshot("创建所有样本后")
    
    # 比较内存
    detector.compare_snapshots(0, -1, top_n=5)
    
    print("\n" + detector.generate_report())
    detector.stop_tracking()


async def test_subprocess_execution():
    """测试3: 子进程执行"""
    print("\n" + "=" * 80)
    print("测试3: 子进程执行与资源清理")
    print("=" * 80)
    
    detector = MemoryLeakDetector()
    detector.start_tracking()
    detector.take_snapshot("开始")
    
    # 创建测试脚本
    script = _ensure_py_file("""
import sys
import time
print("Script started", file=sys.stderr)
for i in range(3):
    print(f"Iteration {i}")
    sys.stdout.flush()
    time.sleep(0.1)
print("Script finished", file=sys.stderr)
""")
    detector.track_temp_file(script)
    
    # 检查初始子进程数
    initial_children = len(psutil.Process().children())
    print(f"\n初始子进程数: {initial_children}")
    
    # 执行多次
    print("\n执行脚本 15 次...")
    for i in range(15):
        result = await _run_py(script)
        current_children = len(psutil.Process().children())
        
        if i % 5 == 0:
            print(f"  迭代 {i}: 子进程={current_children}, "
                  f"返回码={result['return_code']}, "
                  f"stdout行数={len(result['stdout'].splitlines())}")
            detector.take_snapshot(f"执行{i}次后")
            
        # 不保留引用
        del result
        
    gc.collect()
    await asyncio.sleep(0.5)  # 等待子进程完全退出
    
    final_children = len(psutil.Process().children())
    print(f"\n最终子进程数: {final_children}")
    
    if final_children > initial_children:
        print(f"⚠️  警告: 子进程泄漏! 增加了 {final_children - initial_children} 个")
    else:
        print("✓ 子进程正常清理")
        
    detector.take_snapshot("所有执行完成后")
    detector.compare_snapshots(0, -1, top_n=5)
    
    print("\n" + detector.generate_report())
    detector.stop_tracking()


async def test_large_file_processing():
    """测试4: 大文件处理"""
    print("\n" + "=" * 80)
    print("测试4: 大文件处理内存使用")
    print("=" * 80)
    
    detector = MemoryLeakDetector()
    detector.start_tracking()
    detector.take_snapshot("开始")
    
    # 创建大 CSV
    print("\n创建大文件 (10万行)...")
    large_csv = Path(tempfile.gettempdir()) / "large_test.csv"
    with large_csv.open('w') as f:
        f.write("id,value1,value2,value3,text\n")
        for i in range(100000):
            f.write(f"{i},{i*2},{i*3},{i*4},text_{i}\n")
    
    file_size_mb = large_csv.stat().st_size / 1024 / 1024
    print(f"文件大小: {file_size_mb:.2f} MB")
    detector.track_temp_file(large_csv)
    detector.take_snapshot("创建大文件后")
    
    # 测试采样 - 多次
    print("\n创建 5 个样本...")
    for i in range(5):
        sample = _create_debug_sample(large_csv, sample_lines=10)
        detector.track_temp_file(sample)
        print(f"  样本 {i+1}: {sample.stat().st_size / 1024:.2f} KB")
        
        if i % 2 == 0:
            detector.take_snapshot(f"创建样本{i+1}后")
        
        del sample
    
    gc.collect()
    detector.take_snapshot("清理引用后")
    
    # 比较
    detector.compare_snapshots(1, -1, top_n=5)
    
    print("\n" + detector.generate_report())
    detector.stop_tracking()
    
    # 清理
    if large_csv.exists():
        large_csv.unlink()
        print(f"\n已清理大文件")


async def test_state_like_accumulation():
    """测试5: 模拟 state 数据累积"""
    print("\n" + "=" * 80)
    print("测试5: State 数据累积模拟")
    print("=" * 80)
    
    detector = MemoryLeakDetector()
    detector.start_tracking()
    
    # 模拟 state
    state_data = {}
    detector.take_snapshot("空state")
    
    print("\n模拟 100 次操作...")
    for i in range(100):
        # 模拟添加数据
        state_data[f'key_{i}'] = 'x' * 10000  # 10KB
        state_data[f'code_{i}'] = 'def func():\n    pass\n' * 500  # ~15KB
        
        if i % 20 == 0:
            size_kb = sum(sys.getsizeof(v) for v in state_data.values()) / 1024
            print(f"  迭代 {i}: keys={len(state_data)}, 总大小={size_kb:.2f} KB")
            detector.take_snapshot(f"迭代{i}")
    
    detector.take_snapshot("100次迭代后")
    
    # 清理
    print("\n清理 state_data...")
    state_data.clear()
    gc.collect()
    
    detector.take_snapshot("清理后")
    
    detector.compare_snapshots(0, -2, top_n=5)  # 清理前
    detector.compare_snapshots(-2, -1, top_n=5)  # 清理后
    
    print("\n" + detector.generate_report())
    detector.stop_tracking()


# ============================================================================
#                              主入口
# ============================================================================

async def run_all_tests():
    """运行所有测试"""
    print("\n" + "🔍 " * 20)
    print("开始内存泄漏检测")
    print("🔍 " * 20)
    
    tests = [
        ("临时文件创建", test_temp_file_creation),
        ("调试样本创建", test_debug_sample_creation),
        ("子进程执行", test_subprocess_execution),
        ("大文件处理", test_large_file_processing),
        ("State累积", test_state_like_accumulation),
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            result = await test_func()
            results[name] = {"status": "✓ 通过", "result": result}
        except Exception as e:
            print(f"\n❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {"status": "✗ 失败", "error": str(e)}
        
        # 每个测试之间清理
        gc.collect()
        await asyncio.sleep(1)
    
    # 汇总报告
    print("\n" + "=" * 80)
    print("测试汇总")
    print("=" * 80)
    for name, result in results.items():
        print(f"{result['status']} - {name}")
    
    print("\n" + "🔍 " * 20)
    print("所有测试完成!")
    print("🔍 " * 20)


if __name__ == "__main__":
    asyncio.run(run_all_tests())