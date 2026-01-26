-- ==============================================================================
-- Paper2Any Supabase Schema Setup Script
--
-- This script sets up the necessary tables, views, functions, triggers,
-- storage buckets, and security policies for the Paper2Any application.
--
-- INCLUDES:
-- - User management (profiles, referrals, points system)
-- - File storage (user_files, knowledge_base_files)
-- - Usage tracking and quota management
-- - Storage buckets and RLS policies
--
-- INSTRUCTIONS:
-- 1. Go to your Supabase Project Dashboard: https://supabase.com/dashboard
-- 2. Navigate to the "SQL Editor" section.
-- 3. Click "New query", paste this entire script, and click "Run".
--
-- Last Updated: 2026-01-26 (merged with knowledge base schema)
-- ==============================================================================

-- ==============================================================================
-- Table: usage_records
-- Tracks API/Workflow usage for quota management.
-- ==============================================================================
DROP POLICY IF EXISTS "Authenticated users can upload files" ON storage.objects;
DROP POLICY IF EXISTS "Users can view own files" ON storage.objects;
DROP POLICY IF EXISTS "Users can delete own files" ON storage.objects;

CREATE TABLE IF NOT EXISTS public.usage_records (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    workflow_type TEXT NOT NULL,
    called_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE public.usage_records ENABLE ROW LEVEL SECURITY;

-- Policy: Allow users to insert their own usage records
CREATE POLICY "Allow creation of usage records"
ON public.usage_records
FOR INSERT
WITH CHECK (true);

-- Policy: Allow users to view their own usage records
CREATE POLICY "Allow users to view their own usage"
ON public.usage_records
FOR SELECT
USING (auth.uid() = user_id);

-- ==============================================================================
-- Table: user_files
-- Stores metadata for generated files.
-- ==============================================================================

CREATE TABLE IF NOT EXISTS public.user_files (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    file_name TEXT NOT NULL,
    file_size BIGINT,
    workflow_type TEXT,
    file_path TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE public.user_files ENABLE ROW LEVEL SECURITY;

-- Add index for performance
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
-- Table: knowledge_base_files
-- Stores metadata for knowledge base files (PDFs, videos, documents, etc.)
-- ==============================================================================

CREATE TABLE IF NOT EXISTS public.knowledge_base_files (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    user_email TEXT,
    file_name TEXT NOT NULL,
    file_type TEXT,
    file_size BIGINT,
    storage_path TEXT NOT NULL,
    is_embedded BOOLEAN DEFAULT FALSE,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE public.knowledge_base_files ENABLE ROW LEVEL SECURITY;

-- Add index for performance
CREATE INDEX IF NOT EXISTS idx_kb_files_user_id ON public.knowledge_base_files(user_id);

-- Policy: Users can only see their own KB files
CREATE POLICY "Users can view own KB files"
ON public.knowledge_base_files
FOR SELECT
USING (auth.uid() = user_id);

-- Policy: Users can insert their own KB files
CREATE POLICY "Users can insert own KB files"
ON public.knowledge_base_files
FOR INSERT
WITH CHECK (auth.uid() = user_id);

-- Policy: Users can delete their own KB files
CREATE POLICY "Users can delete own KB files"
ON public.knowledge_base_files
FOR DELETE
USING (auth.uid() = user_id);

-- ==============================================================================
-- Table: profiles
-- Stores user profiles with invite codes.
-- ==============================================================================

CREATE TABLE IF NOT EXISTS public.profiles (
    user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    invite_code TEXT UNIQUE NOT NULL DEFAULT upper(substr(md5(random()::text), 1, 8)),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- Policy: Users can view own profile
CREATE POLICY "Users can view own profile"
ON public.profiles
FOR SELECT
USING (auth.uid() = user_id);

-- ==============================================================================
-- Table: referrals
-- Tracks who invited whom.
-- ==============================================================================

CREATE TABLE IF NOT EXISTS public.referrals (
    id BIGSERIAL PRIMARY KEY,
    inviter_user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    invitee_user_id UUID NOT NULL UNIQUE REFERENCES auth.users(id) ON DELETE CASCADE,
    invite_code TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE public.referrals ENABLE ROW LEVEL SECURITY;

-- Policy: Users can view own referrals
CREATE POLICY "Users can view own referrals"
ON public.referrals
FOR SELECT
USING (auth.uid() = inviter_user_id OR auth.uid() = invitee_user_id);

-- ==============================================================================
-- Table: points_ledger
-- Records all points (usage count) transactions.
-- ==============================================================================

CREATE TABLE IF NOT EXISTS public.points_ledger (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    points INTEGER NOT NULL,
    reason TEXT NOT NULL,
    event_key TEXT UNIQUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE public.points_ledger ENABLE ROW LEVEL SECURITY;

-- Policy: Users can view own points
CREATE POLICY "Users can view own points"
ON public.points_ledger
FOR SELECT
USING (auth.uid() = user_id);

-- ==============================================================================
-- View: points_balance
-- Calculates current balance per user.
-- ==============================================================================

CREATE OR REPLACE VIEW public.points_balance AS
SELECT 
    user_id,
    COALESCE(SUM(points), 0)::INTEGER AS balance
FROM public.points_ledger
GROUP BY user_id;

-- ==============================================================================
-- Function: handle_new_user
-- Trigger function to create profile and award signup bonus.
-- ==============================================================================

CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (user_id)
    VALUES (NEW.id)
    ON CONFLICT (user_id) DO NOTHING;
    
    -- Award signup bonus: 20 usage counts
    INSERT INTO public.points_ledger (user_id, points, reason, event_key)
    VALUES (NEW.id, 20, 'signup_bonus', 'signup_bonus_' || NEW.id::text)
    ON CONFLICT (event_key) DO NOTHING;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger: on_auth_user_created
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- ==============================================================================
-- Function: apply_invite_code
-- Claims invite code and awards points to both parties.
-- ==============================================================================

CREATE OR REPLACE FUNCTION public.apply_invite_code(p_code TEXT)
RETURNS JSON AS $$
DECLARE
    v_inviter_id UUID;
    v_invitee_id UUID := auth.uid();
    v_existing_referral BIGINT;
    v_inviter_points INTEGER := 10;
    v_invitee_points INTEGER := 10;
BEGIN
    -- Check if user is logged in
    IF v_invitee_id IS NULL THEN
        RETURN json_build_object('success', false, 'error', 'not_authenticated');
    END IF;

    -- Check if already claimed an invite code
    SELECT id INTO v_existing_referral
    FROM public.referrals
    WHERE invitee_user_id = v_invitee_id;
    
    IF v_existing_referral IS NOT NULL THEN
        RETURN json_build_object('success', false, 'error', 'already_claimed');
    END IF;

    -- Find inviter by invite code
    SELECT user_id INTO v_inviter_id
    FROM public.profiles
    WHERE invite_code = UPPER(p_code);

    IF v_inviter_id IS NULL THEN
        RETURN json_build_object('success', false, 'error', 'invalid_code');
    END IF;

    -- Cannot invite yourself
    IF v_inviter_id = v_invitee_id THEN
        RETURN json_build_object('success', false, 'error', 'self_invite');
    END IF;

    -- Create referral record
    INSERT INTO public.referrals (inviter_user_id, invitee_user_id, invite_code)
    VALUES (v_inviter_id, v_invitee_id, UPPER(p_code));

    -- Award points to inviter
    INSERT INTO public.points_ledger (user_id, points, reason, event_key)
    VALUES (v_inviter_id, v_inviter_points, 'referral_inviter', 
            'referral_inviter_' || v_inviter_id::text || '_' || v_invitee_id::text);

    -- Award points to invitee
    INSERT INTO public.points_ledger (user_id, points, reason, event_key)
    VALUES (v_invitee_id, v_invitee_points, 'referral_invitee', 
            'referral_invitee_' || v_invitee_id::text);

    RETURN json_build_object('success', true, 'inviter_id', v_inviter_id);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

GRANT EXECUTE ON FUNCTION public.apply_invite_code(TEXT) TO authenticated;

-- ==============================================================================
-- Function: deduct_points
-- Deducts points from user balance.
-- ==============================================================================

CREATE OR REPLACE FUNCTION public.deduct_points(
    p_user_id UUID,
    p_amount INTEGER,
    p_reason TEXT
) RETURNS BOOLEAN AS $$
DECLARE
    v_current_balance INTEGER;
    v_event_key TEXT;
BEGIN
    -- Get current balance
    SELECT balance INTO v_current_balance
    FROM public.points_balance
    WHERE user_id = p_user_id;
    
    -- If no balance record exists, user has 0 points
    IF v_current_balance IS NULL THEN
        v_current_balance := 0;
    END IF;
    
    -- Check if user has enough points
    IF v_current_balance < p_amount THEN
        RETURN FALSE;
    END IF;
    
    -- Generate unique event_key using timestamp
    v_event_key := p_reason || '_' || p_user_id::text || '_' || extract(epoch from now())::text;
    
    -- Deduct points by inserting negative ledger entry
    INSERT INTO public.points_ledger (user_id, points, reason, event_key)
    VALUES (p_user_id, -p_amount, p_reason, v_event_key);
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

GRANT EXECUTE ON FUNCTION public.deduct_points(UUID, INTEGER, TEXT) TO authenticated;

-- ==============================================================================
-- Function: check_and_grant_daily_usage
-- Grants 10 daily usage counts if user balance <= 30.
-- ==============================================================================

CREATE OR REPLACE FUNCTION public.check_and_grant_daily_usage(p_user_id UUID)
RETURNS INTEGER AS $$
DECLARE
    v_balance INTEGER;
    v_event_key TEXT;
BEGIN
    -- Get current balance from view
    SELECT balance INTO v_balance
    FROM public.points_balance
    WHERE user_id = p_user_id;
    
    -- If no balance record exists, user has 0 points
    IF v_balance IS NULL THEN
        v_balance := 0;
    END IF;
    
    -- Check if balance > 30, no daily grant
    IF v_balance > 30 THEN
        RETURN v_balance;
    END IF;
    
    -- Generate event_key for today's grant (idempotency)
    v_event_key := 'daily_grant_' || CURRENT_DATE::text || '_' || p_user_id::text;
    
    -- Grant 10 usage counts (idempotent insert using event_key)
    INSERT INTO public.points_ledger (user_id, points, reason, event_key)
    VALUES (p_user_id, 10, 'daily_grant', v_event_key)
    ON CONFLICT (event_key) DO NOTHING;
    
    -- Return new balance (recalculate from view)
    SELECT balance INTO v_balance
    FROM public.points_balance
    WHERE user_id = p_user_id;
    
    RETURN COALESCE(v_balance, 0);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

GRANT EXECUTE ON FUNCTION public.check_and_grant_daily_usage(UUID) TO authenticated;

COMMENT ON FUNCTION public.check_and_grant_daily_usage IS 
'Grants 10 daily usage counts if user balance <= 30. Idempotent - safe to call multiple times per day.';

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
