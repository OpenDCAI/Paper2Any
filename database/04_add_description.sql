-- Add description column to knowledge_base_files table
ALTER TABLE public.knowledge_base_files 
ADD COLUMN IF NOT EXISTS description TEXT;
