-- Migration: 003_storage_policies.sql
-- Description: Create storage bucket and RLS policies for user files
-- Created: 2025-12-27
--
-- IMPORTANT: First create the bucket in Supabase Dashboard > Storage
-- Bucket name: user-files (NOT public)
-- Then run these policies in SQL Editor

-- ============================================
-- Storage Policies for 'user-files' bucket
-- ============================================
-- File path convention: user-files/{user_id}/{timestamp}_{filename}
-- Example: user-files/550e8400-e29b-41d4-a716-446655440000/20251227_143052_report.pdf

-- Users can upload files to their own folder
CREATE POLICY "Users upload own files"
    ON storage.objects
    FOR INSERT
    WITH CHECK (
        bucket_id = 'user-files' AND
        auth.uid()::text = (storage.foldername(name))[1]
    );

-- Users can read/download their own files
CREATE POLICY "Users read own files"
    ON storage.objects
    FOR SELECT
    USING (
        bucket_id = 'user-files' AND
        auth.uid()::text = (storage.foldername(name))[1]
    );

-- Users can update their own files (e.g., replace)
CREATE POLICY "Users update own files"
    ON storage.objects
    FOR UPDATE
    USING (
        bucket_id = 'user-files' AND
        auth.uid()::text = (storage.foldername(name))[1]
    );

-- Users can delete their own files
CREATE POLICY "Users delete own files"
    ON storage.objects
    FOR DELETE
    USING (
        bucket_id = 'user-files' AND
        auth.uid()::text = (storage.foldername(name))[1]
    );

-- ============================================
-- How storage.foldername() works
-- ============================================
-- storage.foldername('550e8400.../file.pdf') returns ARRAY['550e8400...']
-- We compare [1] (first element) with auth.uid() to verify ownership
-- This prevents User A from accessing User B's folder
