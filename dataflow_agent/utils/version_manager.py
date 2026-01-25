from pathlib import Path
from typing import List, Optional, Tuple
import re
import time
import shutil
import json
import os


class ImageVersionManager:
    """管理幻灯片图片的版本化存储"""

    MAX_VERSIONS = int(os.getenv("MAX_IMAGE_VERSIONS", "10"))  # 可通过环境变量配置

    @staticmethod
    def get_next_version_number(img_dir: Path, page_idx: int) -> int:
        """扫描目录中的现有版本并返回下一个版本号"""
        pattern = f"page_{page_idx:03d}_v*.png"
        existing = list(img_dir.glob(pattern))
        if not existing:
            return 1

        version_nums = []
        for f in existing:
            match = re.search(r'_v(\d+)\.png$', f.name)
            if match:
                version_nums.append(int(match.group(1)))

        return max(version_nums) + 1 if version_nums else 1

    @staticmethod
    def save_versioned_image(
        img_dir: Path,
        page_idx: int,
        new_image_path: str,
        prompt: str = ""
    ) -> Tuple[str, int]:
        """
        保存新版本并更新当前指针。

        Args:
            img_dir: 图片目录路径
            page_idx: 页面索引
            new_image_path: 新图片的路径
            prompt: 用户的编辑提示词

        Returns:
            (versioned_path, version_number): 版本化路径和版本号的元组
        """
        # 检查是否是第一次编辑（需要保留原始版本）
        current_path = img_dir / f"page_{page_idx:03d}.png"
        version_num = ImageVersionManager.get_next_version_number(img_dir, page_idx)

        # 特殊情况：如果这是版本 1，先将当前图片保存为 v001
        if version_num == 1 and current_path.exists():
            v001_path = img_dir / f"page_{page_idx:03d}_v001.png"
            shutil.copy2(current_path, v001_path)
            # 保存原始版本的元数据
            ImageVersionManager._save_version_metadata(
                img_dir, page_idx, 1, "Initial generation"
            )
            version_num = 2  # 新编辑成为 v002

        # 保存为版本化文件
        versioned_name = f"page_{page_idx:03d}_v{version_num:03d}.png"
        versioned_path = img_dir / versioned_name
        shutil.copy2(new_image_path, versioned_path)

        # 更新当前指针（复制，而不是符号链接，以兼容 Windows）
        shutil.copy2(new_image_path, current_path)

        # 清理超过限制的旧版本
        ImageVersionManager._cleanup_old_versions(img_dir, page_idx)

        # 保存元数据
        ImageVersionManager._save_version_metadata(
            img_dir, page_idx, version_num, prompt
        )

        return str(versioned_path), version_num

    @staticmethod
    def _cleanup_old_versions(img_dir: Path, page_idx: int):
        """删除超过 MAX_VERSIONS 限制的版本"""
        pattern = f"page_{page_idx:03d}_v*.png"
        versions = sorted(img_dir.glob(pattern))

        if len(versions) > ImageVersionManager.MAX_VERSIONS:
            to_delete = versions[:-ImageVersionManager.MAX_VERSIONS]
            for old_file in to_delete:
                old_file.unlink()
                # 同时删除对应的元数据
                meta_file = old_file.with_suffix('.json')
                if meta_file.exists():
                    meta_file.unlink()

    @staticmethod
    def _save_version_metadata(
        img_dir: Path,
        page_idx: int,
        version_num: int,
        prompt: str
    ):
        """保存此版本的元数据 JSON"""
        meta_file = img_dir / f"page_{page_idx:03d}_v{version_num:03d}.json"
        metadata = {
            "version": version_num,
            "page_index": page_idx,
            "prompt": prompt,
            "timestamp": int(time.time())
        }
        meta_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

    @staticmethod
    def get_version_history(img_dir: Path, page_idx: int) -> List[dict]:
        """检索页面的所有版本及元数据"""
        pattern = f"page_{page_idx:03d}_v*.png"
        versions = sorted(img_dir.glob(pattern))

        history = []
        for img_file in versions:
            meta_file = img_file.with_suffix('.json')
            if meta_file.exists():
                try:
                    metadata = json.loads(meta_file.read_text())
                except json.JSONDecodeError:
                    # 如果元数据缺失或损坏，使用回退方案
                    match = re.search(r'_v(\d+)\.png$', img_file.name)
                    version_num = int(match.group(1)) if match else 0
                    metadata = {
                        "version": version_num,
                        "page_index": page_idx,
                        "prompt": "",
                        "timestamp": int(img_file.stat().st_mtime)
                    }
            else:
                # 如果元数据缺失，使用回退方案
                match = re.search(r'_v(\d+)\.png$', img_file.name)
                version_num = int(match.group(1)) if match else 0
                metadata = {
                    "version": version_num,
                    "page_index": page_idx,
                    "prompt": "",
                    "timestamp": int(img_file.stat().st_mtime)
                }

            metadata["image_path"] = str(img_file)
            history.append(metadata)

        return history

    @staticmethod
    def revert_to_version(
        img_dir: Path,
        page_idx: int,
        target_version: int
    ) -> Optional[str]:
        """将当前图片恢复到特定版本"""
        versioned_file = img_dir / f"page_{page_idx:03d}_v{target_version:03d}.png"

        if not versioned_file.exists():
            return None

        current_path = img_dir / f"page_{page_idx:03d}.png"
        shutil.copy2(versioned_file, current_path)

        return str(current_path)
