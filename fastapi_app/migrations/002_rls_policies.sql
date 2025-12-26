-- Migration: 002_rls_policies.sql
-- Description: Enable Row Level Security and create access policies
-- Created: 2025-12-27
--
-- IMPORTANT: Run this AFTER 001_initial_schema.sql
-- RLS ensures users can only access their own data

-- ============================================
-- Enable Row Level Security
-- ============================================
ALTER TABLE usage_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_files ENABLE ROW LEVEL SECURITY;

-- ============================================
-- Policies for usage_records
-- ============================================

-- Users can read their own usage records
CREATE POLICY "Users read own usage"
    ON usage_records
    FOR SELECT
    USING (auth.uid() = user_id);

-- Users can insert their own usage records
CREATE POLICY "Users insert own usage"
    ON usage_records
    FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- Note: No UPDATE or DELETE policies - usage records are immutable

-- ============================================
-- Policies for user_files
-- ============================================

-- Users have full access to their own files (SELECT, INSERT, UPDATE, DELETE)
CREATE POLICY "Users manage own files"
    ON user_files
    FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- ============================================
-- Service role bypass note
-- ============================================
-- The service_role key bypasses RLS completely.
-- Use it ONLY in backend server code, never expose to clients.
-- The anon key respects RLS and is safe for client-side use.
