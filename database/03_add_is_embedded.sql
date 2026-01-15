-- Add is_embedded column to knowledge_base_files table
ALTER TABLE public.knowledge_base_files 
ADD COLUMN IF NOT EXISTS is_embedded BOOLEAN DEFAULT FALSE;
