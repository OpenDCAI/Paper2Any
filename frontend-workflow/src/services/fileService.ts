/**
 * File service for saving workflow output files to Supabase Storage.
 *
 * Uploads files to Storage and saves metadata to user_files table.
 */

import { supabase, isSupabaseConfigured } from "../lib/supabase";

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

    // Generate unique file path: user_id/timestamp_filename
    const timestamp = Date.now();
    const filePath = `${user.id}/${timestamp}_${fileName}`;

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
  if (!isSupabaseConfigured()) {
    return [];
  }

  try {
    const { data, error } = await supabase
      .from("user_files")
      .select("*")
      .order("created_at", { ascending: false });

    if (error) {
      console.error("[fileService] Failed to get file records:", error);
      return [];
    }

    return (data || []).map((row) => ({
      id: row.id,
      file_name: row.file_name,
      file_size: row.file_size,
      workflow_type: row.workflow_type,
      created_at: row.created_at,
      download_url: row.file_path || undefined,
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
  if (!isSupabaseConfigured()) {
    return false;
  }

  try {
    // First get the file record to find the storage path
    const { data: record, error: fetchError } = await supabase
      .from("user_files")
      .select("file_path")
      .eq("id", fileId)
      .single();

    if (fetchError) {
      console.error("[fileService] Failed to fetch file record:", fetchError);
      return false;
    }

    // Delete the file from Storage if it exists
    if (record?.file_path) {
      try {
        // Extract path from URL: https://xxx.supabase.co/storage/v1/object/public/user-files/user_id/filename
        const url = new URL(record.file_path);
        const pathMatch = url.pathname.match(/\/storage\/v1\/object\/public\/user-files\/(.+)/);
        if (pathMatch?.[1]) {
          await supabase.storage.from(STORAGE_BUCKET).remove([decodeURIComponent(pathMatch[1])]);
        }
      } catch (e) {
        console.warn("[fileService] Failed to delete file from storage:", e);
        // Continue to delete the record even if storage deletion fails
      }
    }

    // Delete the record from user_files table
    const { error } = await supabase
      .from("user_files")
      .delete()
      .eq("id", fileId);

    if (error) {
      console.error("[fileService] Failed to delete file record:", error);
      return false;
    }

    return true;
  } catch (err) {
    console.error("[fileService] Error deleting file record:", err);
    return false;
  }
}
