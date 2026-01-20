/**
 * File service for saving workflow output files to Supabase Storage.
 *
 * Uploads files to Storage and saves metadata to user_files table.
 */

import { supabase } from "../lib/supabase";
import { API_KEY } from "../config/api";

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
  try {
    // Get JWT token from Supabase (if configured)
    const { data: { session } } = await supabase.auth.getSession();

    // Sanitize filename to avoid special characters
    const sanitizedFileName = sanitizeFileName(fileName, workflowType);
    console.log(`[fileService] Original filename: ${fileName}`);
    console.log(`[fileService] Sanitized filename: ${sanitizedFileName}`);

    // Create FormData for file upload
    const formData = new FormData();
    formData.append('file', blob, sanitizedFileName);
    formData.append('workflow_type', workflowType);

    // Add email if no session (fallback for non-Supabase deployments)
    if (!session) {
      console.warn("[fileService] No authenticated session, using email fallback");
      // Try to get user identifier from user state (if available)
      const user = (window as any).__user_email;
      if (user) {
        formData.append('email', user);
      }
    }

    // Prepare headers
    const headers: Record<string, string> = {
      'X-API-Key': API_KEY,
    };

    // Add JWT token if available
    if (session?.access_token) {
      headers['Authorization'] = `Bearer ${session.access_token}`;
    }

    // Upload to backend
    const response = await fetch('/api/v1/files/upload', {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("[fileService] Failed to upload file:", errorText);
      return null;
    }

    const data = await response.json();

    if (!data.success) {
      console.error("[fileService] Upload failed:", data);
      return null;
    }

    return {
      id: data.file_path, // Use file path as ID
      file_name: data.file_name,
      file_size: data.file_size,
      workflow_type: data.workflow_type,
      created_at: data.created_at,
      download_url: data.file_path, // Backend will serve via /outputs
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
    // Get JWT token from Supabase (if configured)
    const { data: { session } } = await supabase.auth.getSession();

    // Prepare headers
    const headers: Record<string, string> = {
      'X-API-Key': API_KEY,
    };

    // Add JWT token if available
    if (session?.access_token) {
      headers['Authorization'] = `Bearer ${session.access_token}`;
    }

    // Build URL with email parameter if no session
    let url = '/api/v1/files/history';
    if (!session) {
      console.warn("[fileService] No authenticated session, using email fallback");
      const user = (window as any).__user_email;
      if (user) {
        url += `?email=${encodeURIComponent(user)}`;
      }
    }

    // Use unified backend API
    const res = await fetch(url, {
      headers,
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
