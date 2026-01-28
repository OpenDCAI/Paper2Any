-- ==============================================================================
-- Paper2Any Database Cleanup Script
--
-- This script removes all existing tables, functions, triggers, views,
-- and storage policies to prepare for a fresh initialization.
--
-- INSTRUCTIONS:
-- 1. Go to your Supabase Project Dashboard SQL Editor
-- 2. Run this script FIRST to clean up existing objects
-- 3. Then run 01_init_schema.sql to recreate everything
--
-- WARNING: This will DELETE ALL DATA in these tables!
-- ==============================================================================

-- ==============================================================================
-- Step 1: Drop Storage Policies
-- ==============================================================================

DROP POLICY IF EXISTS "Authenticated users can upload files" ON storage.objects;
DROP POLICY IF EXISTS "Users can view own files" ON storage.objects;
DROP POLICY IF EXISTS "Users can delete own files" ON storage.objects;

-- ==============================================================================
-- Step 2: Drop Triggers
-- ==============================================================================

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;

-- ==============================================================================
-- Step 3: Drop Functions
-- ==============================================================================

DROP FUNCTION IF EXISTS public.handle_new_user() CASCADE;
DROP FUNCTION IF EXISTS public.apply_invite_code(TEXT) CASCADE;
DROP FUNCTION IF EXISTS public.deduct_points(UUID, INTEGER, TEXT) CASCADE;
DROP FUNCTION IF EXISTS public.check_and_grant_daily_usage(UUID) CASCADE;

-- ==============================================================================
-- Step 4: Drop Views
-- ==============================================================================

DROP VIEW IF EXISTS public.points_balance CASCADE;

-- ==============================================================================
-- Step 5: Drop Tables (CASCADE will remove all dependent objects)
-- ==============================================================================

DROP TABLE IF EXISTS public.usage_records CASCADE;
DROP TABLE IF EXISTS public.user_files CASCADE;
DROP TABLE IF EXISTS public.knowledge_base_files CASCADE;
DROP TABLE IF EXISTS public.referrals CASCADE;
DROP TABLE IF EXISTS public.points_ledger CASCADE;
DROP TABLE IF EXISTS public.profiles CASCADE;

-- ==============================================================================
-- Step 6: Delete Storage Objects and Bucket
-- ==============================================================================

-- First, delete all objects in the bucket
DELETE FROM storage.objects WHERE bucket_id = 'user-files';

-- Then, delete the bucket itself
DELETE FROM storage.buckets WHERE id = 'user-files';

-- ==============================================================================
-- Cleanup Complete!
-- Now you can run 01_init_schema.sql to recreate everything fresh.
-- ==============================================================================

