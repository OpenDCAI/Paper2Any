-- ==============================================================================
-- Paper2Any Supabase Schema Setup Script
--
-- This script sets up the necessary tables, storage buckets, and security policies
-- for the Paper2Any application.
--
-- INSTRUCTIONS:
-- 1. Go to your Supabase Project Dashboard: https://supabase.com/dashboard
-- 2. Navigate to the "SQL Editor" section.
-- 3. Click "New query", paste this entire script, and click "Run".
-- ==============================================================================

-- 1. Enable RLS (Row Level Security) on public schema just in case
-- (Tables created below will have RLS enabled by default or explicitly)

-- ==============================================================================
-- Table: usage_records
-- Tracks API/Workflow usage for quota management.
-- ==============================================================================

CREATE TABLE IF NOT EXISTS public.usage_records (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id TEXT NOT NULL, -- Can be UUID from auth.users or a string fingerprint for anon
    workflow_type TEXT NOT NULL,
    called_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Enable Row Level Security
ALTER TABLE public.usage_records ENABLE ROW LEVEL SECURITY;

-- Policy: Allow users to insert their own usage records
-- (Note: For anonymous users, the logic is handled by the application sending a fingerprint.
--  Ideally, for strict security, insert should be server-side, but this app client-side inserts.)
CREATE POLICY "Allow creation of usage records"
ON public.usage_records
FOR INSERT
WITH CHECK (true); -- Ideally restrict to authenticated or validated users

-- Policy: Allow users to view their own usage records (for quota checking)
CREATE POLICY "Allow users to view their own usage"
ON public.usage_records
FOR SELECT
USING (
    -- If authenticated, match auth.uid
    (auth.uid() IS NOT NULL AND user_id = auth.uid()::text)
    OR
    -- If anonymous/public, we rely on the client querying by their ID.
    -- WARNING: This is permissive. For higher security, use a server-side function.
    true
);

-- ==============================================================================
-- Table: user_files
-- Stores metadata for generated files.
-- ==============================================================================

CREATE TABLE IF NOT EXISTS public.user_files (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE, -- Link to Supabase Auth
    file_name TEXT NOT NULL,
    file_size BIGINT,
    workflow_type TEXT,
    file_path TEXT, -- Stores the public URL or storage path
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE public.user_files ENABLE ROW LEVEL SECURITY;

-- Add index for performance (Added in restructuring)
CREATE INDEX IF NOT EXISTS idx_user_files_user_id ON public.user_files(user_id);

-- Policy: Users can only see their own files
CREATE POLICY "Users can view own files"
ON public.user_files
FOR SELECT
USING (auth.uid() = user_id);

-- Policy: Users can insert their own files
CREATE POLICY "Users can upload own files"
ON public.user_files
FOR INSERT
WITH CHECK (auth.uid() = user_id);

-- Policy: Users can delete their own files
CREATE POLICY "Users can delete own files"
ON public.user_files
FOR DELETE
USING (auth.uid() = user_id);

-- ==============================================================================
-- Storage Bucket: user-files
-- Stores the actual binary files (PDFs, PPTs, Images).
-- ==============================================================================

-- Create the bucket if it doesn't exist
INSERT INTO storage.buckets (id, name, public)
VALUES ('user-files', 'user-files', true)
ON CONFLICT (id) DO NOTHING;

-- Policy: Allow authenticated users to upload files to their own folder
CREATE POLICY "Authenticated users can upload files"
ON storage.objects
FOR INSERT
TO authenticated
WITH CHECK (
    bucket_id = 'user-files' AND
    (storage.foldername(name))[1] = auth.uid()::text
);

-- Policy: Users can view/download their own files
-- Since the bucket is public, anyone with the link *can* technically download if they guess it,
-- but this policy governs "SELECT" permission via the API.
CREATE POLICY "Users can view own files"
ON storage.objects
FOR SELECT
TO authenticated
USING (
    bucket_id = 'user-files' AND
    (storage.foldername(name))[1] = auth.uid()::text
);

-- Policy: Users can delete their own files
CREATE POLICY "Users can delete own files"
ON storage.objects
FOR DELETE
TO authenticated
USING (
    bucket_id = 'user-files' AND
    (storage.foldername(name))[1] = auth.uid()::text
);

-- ==============================================================================
-- Done!
-- ==============================================================================
