/**
 * File service for saving workflow output records to Supabase.
 *
 * Directly inserts into user_files table using RLS (user can only access own files).
 */

import { supabase, isSupabaseConfigured } from "../lib/supabase";

export interface FileRecord {
  id?: string;
  file_name: string;
  file_size?: number;
  workflow_type: string;
  created_at?: string;
  download_url?: string;
}

/**
 * Save a file record to Supabase user_files table.
 *
 * Uses RLS - user must be authenticated and can only insert their own records.
 *
 * @param fileName - Name of the generated file (e.g., "output.pptx")
 * @param workflowType - Type of workflow (e.g., "paper2figure", "paper2ppt")
 * @param fileSize - Optional file size in bytes
 * @param downloadUrl - Optional download URL for the file
 * @returns The created file record, or null if failed
 */
export async function saveFileRecord(
  fileName: string,
  workflowType: string,
  fileSize?: number,
  downloadUrl?: string
): Promise<FileRecord | null> {
  if (!isSupabaseConfigured()) {
    console.warn("[fileService] Supabase not configured, skipping file save");
    return null;
  }

  try {
    // Get current user
    const {
      data: { user },
    } = await supabase.auth.getUser();

    if (!user) {
      console.warn("[fileService] No authenticated user, skipping file save");
      return null;
    }

    // Insert file record - RLS ensures user_id matches authenticated user
    const { data, error } = await supabase
      .from("user_files")
      .insert({
        user_id: user.id,
        file_name: fileName,
        file_size: fileSize || null,
        workflow_type: workflowType,
        file_path: downloadUrl || "", // Store download URL in file_path field
      })
      .select()
      .single();

    if (error) {
      console.error("[fileService] Failed to save file record:", error);
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
    console.error("[fileService] Error saving file record:", err);
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
 * Delete a file record.
 *
 * @param fileId - The file record ID to delete
 * @returns true if deleted, false otherwise
 */
export async function deleteFileRecord(fileId: string): Promise<boolean> {
  if (!isSupabaseConfigured()) {
    return false;
  }

  try {
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
