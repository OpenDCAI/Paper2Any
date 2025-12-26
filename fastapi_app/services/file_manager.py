"""
File management service for user-generated files.

Handles uploading to Supabase Storage and tracking in database.

Usage:
    from fastapi_app.services.file_manager import file_manager, UserFile

    # Upload a file after workflow completion
    file = await file_manager.upload(user_id, "/path/to/output.pptx", "paper2ppt")

    # List user's files
    files = await file_manager.list_files(user_id)

    # Delete a file
    success = await file_manager.delete_file(user_id, file_id)
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from fastapi_app.supabase_client import get_supabase_admin


class UserFile(BaseModel):
    """User file metadata with optional download URL."""

    id: str
    file_name: str
    file_size: Optional[int] = None
    workflow_type: Optional[str] = None
    created_at: str
    download_url: Optional[str] = None


class FileManager:
    """
    Service for managing user files in Supabase Storage.

    Files are stored with path convention: {user_id}/{timestamp}_{filename}
    This enables RLS policies to restrict access per user.
    """

    BUCKET = "user-files"
    SIGNED_URL_EXPIRY = 3600  # 1 hour

    async def upload(
        self, user_id: str, local_path: str, workflow_type: str
    ) -> UserFile:
        """
        Upload a file to Supabase Storage and create database record.

        Args:
            user_id: The user's UUID
            local_path: Local filesystem path to the file
            workflow_type: Type of workflow that generated this file

        Returns:
            UserFile with metadata and (no download_url yet)

        Raises:
            FileNotFoundError: If local_path doesn't exist
            Exception: On upload or database error
        """
        supabase = get_supabase_admin()
        path = Path(local_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")

        # Generate storage path: {user_id}/{timestamp}_{filename}
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        storage_path = f"{user_id}/{ts}_{path.name}"

        # Upload to Supabase Storage
        with open(local_path, "rb") as f:
            file_content = f.read()

        supabase.storage.from_(self.BUCKET).upload(
            storage_path,
            file_content,
            file_options={"content-type": self._guess_content_type(path.name)},
        )

        # Insert database record
        record = {
            "user_id": user_id,
            "file_path": storage_path,
            "file_name": path.name,
            "file_size": path.stat().st_size,
            "workflow_type": workflow_type,
        }
        result = supabase.table("user_files").insert(record).execute()
        row = result.data[0]

        return UserFile(
            id=row["id"],
            file_name=row["file_name"],
            file_size=row["file_size"],
            workflow_type=row["workflow_type"],
            created_at=row["created_at"],
        )

    async def list_files(self, user_id: str) -> List[UserFile]:
        """
        List all files for a user with signed download URLs.

        Args:
            user_id: The user's UUID

        Returns:
            List of UserFile objects sorted by creation date (newest first)
        """
        supabase = get_supabase_admin()

        result = (
            supabase.table("user_files")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )

        files = []
        for row in result.data:
            # Generate signed URL for download
            download_url = None
            try:
                url_result = supabase.storage.from_(self.BUCKET).create_signed_url(
                    row["file_path"], self.SIGNED_URL_EXPIRY
                )
                download_url = url_result.get("signedURL")
            except Exception:
                pass  # File might have been deleted from storage

            files.append(
                UserFile(
                    id=row["id"],
                    file_name=row["file_name"],
                    file_size=row["file_size"],
                    workflow_type=row["workflow_type"],
                    created_at=row["created_at"],
                    download_url=download_url,
                )
            )

        return files

    async def delete_file(self, user_id: str, file_id: str) -> bool:
        """
        Delete a file from both storage and database.

        Only deletes if the file belongs to the specified user.

        Args:
            user_id: The user's UUID (for ownership verification)
            file_id: The file record UUID

        Returns:
            True if deleted, False if not found or not owned by user
        """
        supabase = get_supabase_admin()

        # Get file path (with ownership check)
        result = (
            supabase.table("user_files")
            .select("file_path")
            .eq("id", file_id)
            .eq("user_id", user_id)
            .execute()
        )

        if not result.data:
            return False

        file_path = result.data[0]["file_path"]

        # Delete from storage
        try:
            supabase.storage.from_(self.BUCKET).remove([file_path])
        except Exception:
            pass  # File might already be deleted from storage

        # Delete from database
        supabase.table("user_files").delete().eq("id", file_id).execute()

        return True

    async def get_file(self, user_id: str, file_id: str) -> Optional[UserFile]:
        """
        Get a single file's metadata with download URL.

        Args:
            user_id: The user's UUID (for ownership verification)
            file_id: The file record UUID

        Returns:
            UserFile if found and owned by user, None otherwise
        """
        supabase = get_supabase_admin()

        result = (
            supabase.table("user_files")
            .select("*")
            .eq("id", file_id)
            .eq("user_id", user_id)
            .execute()
        )

        if not result.data:
            return None

        row = result.data[0]

        # Generate signed URL
        download_url = None
        try:
            url_result = supabase.storage.from_(self.BUCKET).create_signed_url(
                row["file_path"], self.SIGNED_URL_EXPIRY
            )
            download_url = url_result.get("signedURL")
        except Exception:
            pass

        return UserFile(
            id=row["id"],
            file_name=row["file_name"],
            file_size=row["file_size"],
            workflow_type=row["workflow_type"],
            created_at=row["created_at"],
            download_url=download_url,
        )

    def _guess_content_type(self, filename: str) -> str:
        """Guess content type from filename extension."""
        ext = Path(filename).suffix.lower()
        content_types = {
            ".pdf": "application/pdf",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".ppt": "application/vnd.ms-powerpoint",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
            ".mp4": "video/mp4",
            ".json": "application/json",
            ".txt": "text/plain",
        }
        return content_types.get(ext, "application/octet-stream")


# Module-level singleton
file_manager = FileManager()
