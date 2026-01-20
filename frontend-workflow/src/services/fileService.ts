/**
 * File service for saving workflow output files to Supabase Storage.
 *
 * Uploads files to Storage and saves metadata to user_files table.
 */

import { supabase, isSupabaseConfigured } from "../lib/supabase";
import { useAuthStore } from "../stores/authStore";
import { API_KEY } from "../config/api";

const STORAGE_BUCKET = "user-files";

export interface FileRecord {
  id?: string;
  file_name: string;
  file_size?: number;
  workflow_type: string;
  created_at?: string;
  download_url?: string;
}

/**
 * Upload a file to Supabase Storage and save record to user_files table.
 *
 * @param blob - The file blob to upload
 * @param fileName - Name of the file
 * @param workflowType - Type of workflow that generated this file
 * @returns The created file record with download URL, or null if failed
 */
/**
 * Sanitize filename to be compatible with Supabase Storage.
 * Removes or replaces characters that are not allowed in storage keys.
 * If the filename becomes empty after sanitization (e.g., all Chinese characters),
 * uses a fallback name with timestamp.
 */
function sanitizeFileName(fileName: string, workflowType: string): string {
  // Get file extension
  const lastDotIndex = fileName.lastIndexOf('.');
  const name = lastDotIndex > 0 ? fileName.substring(0, lastDotIndex) : fileName;
  const ext = lastDotIndex > 0 ? fileName.substring(lastDotIndex) : '';

  // Replace spaces with underscores
  // Remove or replace special characters and non-ASCII characters
  // Keep only: alphanumeric, underscore, hyphen, dot
  const sanitized = name
    .replace(/\s+/g, '_')  // Replace spaces with underscores
    .replace(/[^\w\-\.]/g, '')  // Remove non-alphanumeric except underscore, hyphen, dot
    .substring(0, 100);  // Limit length to 100 chars

  // If sanitized name is empty (all non-ASCII chars removed), use fallback
  if (!sanitized || sanitized.trim() === '') {
    const timestamp = Date.now();
    return `${workflowType}_${timestamp}${ext}`;
  }

  return sanitized + ext;
}

export async function uploadAndSaveFile(
  blob: Blob,
  fileName: string,
  workflowType: string
): Promise<FileRecord | null> {
  if (!isSupabaseConfigured()) {
    console.warn("[fileService] Supabase not configured, skipping file upload");
    return null;
  }

  try {
    const { data: { user } } = await supabase.auth.getUser();

    if (!user) {
      console.warn("[fileService] No authenticated user, skipping file upload");
      return null;
    }

    // Sanitize filename to avoid Supabase Storage errors
    const sanitizedFileName = sanitizeFileName(fileName, workflowType);
    console.log(`[fileService] Original filename: ${fileName}`);
    console.log(`[fileService] Sanitized filename: ${sanitizedFileName}`);

    // Generate unique file path: user_id/timestamp_filename
    const timestamp = Date.now();
    const filePath = `${user.id}/${timestamp}_${sanitizedFileName}`;

    // Upload to Supabase Storage
    const { error: uploadError } = await supabase.storage
      .from(STORAGE_BUCKET)
      .upload(filePath, blob, {
        contentType: blob.type || "application/octet-stream",
        upsert: false,
      });

    if (uploadError) {
      console.error("[fileService] Failed to upload file:", uploadError);
      return null;
    }

    // Get public URL
    const { data: urlData } = supabase.storage
      .from(STORAGE_BUCKET)
      .getPublicUrl(filePath);

    const downloadUrl = urlData.publicUrl;

    // Save record to user_files table
    const { data, error } = await supabase
      .from("user_files")
      .insert({
        user_id: user.id,
        file_name: fileName,
        file_size: blob.size,
        workflow_type: workflowType,
        file_path: downloadUrl,
      })
      .select()
      .single();

    if (error) {
      console.error("[fileService] Failed to save file record:", error);
      // Try to delete uploaded file on failure
      await supabase.storage.from(STORAGE_BUCKET).remove([filePath]);
      return null;
    }

    return {
      id: data.id,
      file_name: data.file_name,
      file_size: data.file_size,
      workflow_type: data.workflow_type,
      created_at: data.created_at,
      download_url: downloadUrl,
    };
  } catch (err) {
    console.error("[fileService] Error uploading file:", err);
    return null;
  }
}

/**
 * Get all file records for the current user.
 *
 * @returns List of file records sorted by created_at desc
 */
export async function getFileRecords(): Promise<FileRecord[]> {
  try {
    const user = useAuthStore.getState().user;
    const email = user?.email;

    // 如果有 email，使用后端接口获取本地历史文件
    if (email) {
      const res = await fetch(`/api/v1/paper2figure/history?email=${encodeURIComponent(email)}`, {
        headers: {
          'X-API-Key': API_KEY,
        },
      });
      if (!res.ok) {
        console.error(`[fileService] History API failed: ${res.statusText}`);
        return [];
      }
      
      const data = await res.json();
      if (!data.success) {
        console.error("[fileService] History API returned error", data);
        return [];
      }
      
      return data.files || [];
    }

    // 如果没有 email（手机号登录），从 Supabase user_files 表查询
    if (!isSupabaseConfigured() || !user) {
      return [];
    }

    const { data, error } = await supabase
      .from("user_files")
      .select("*")
      .eq("user_id", user.id)
      .order("created_at", { ascending: false });

    if (error) {
      console.error("[fileService] Failed to fetch user_files:", error);
      return [];
    }

    return (data || []).map((record) => ({
      id: record.id,
      file_name: record.file_name,
      file_size: record.file_size,
      workflow_type: record.workflow_type,
      created_at: record.created_at,
      download_url: record.file_path,
    }));

  } catch (err) {
    console.error("[fileService] Error getting file records:", err);
    return [];
  }
}

/**
 * Delete a file record and its associated file from Storage.
 *
 * @param fileId - The file record ID to delete
 * @returns true if deleted, false otherwise
 */
export async function deleteFileRecord(fileId: string): Promise<boolean> {
  // 目前本地文件删除接口尚未实现
  console.warn("[fileService] Local file deletion not implemented yet.");
  return false;
  
  /* 原 Supabase 删除逻辑暂存
  if (!isSupabaseConfigured()) {
    return false;
  }

  try {
    // ... (original code)
  } catch (err) {
    console.error("[fileService] Error deleting file record:", err);
    return false;
  }
  */
}
