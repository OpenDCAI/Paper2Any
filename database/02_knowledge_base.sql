-- ==============================================================================
-- Knowledge Base Schema Update
-- Run this script in Supabase SQL Editor to add the KB table
-- ==============================================================================

CREATE TABLE IF NOT EXISTS public.knowledge_base_files (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    user_email TEXT, -- Stored for convenience in directory mapping
    file_name TEXT NOT NULL,
    file_type TEXT, -- e.g., 'pdf', 'mp4', 'docx'
    file_size BIGINT,
    storage_path TEXT NOT NULL, -- Relative path or full path on server
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.knowledge_base_files ENABLE ROW LEVEL SECURITY;

-- Policies
CREATE POLICY "Users can view own KB files"
ON public.knowledge_base_files
FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own KB files"
ON public.knowledge_base_files
FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own KB files"
ON public.knowledge_base_files
FOR DELETE
USING (auth.uid() = user_id);

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_kb_files_user_id ON public.knowledge_base_files(user_id);
