-- Invite and Points System Migration
-- Creates profiles, referrals, points_ledger tables and apply_invite_code RPC

-- 1. profiles table: stores user invite codes
CREATE TABLE IF NOT EXISTS public.profiles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  invite_code TEXT UNIQUE NOT NULL DEFAULT upper(substr(md5(random()::text), 1, 8)),
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now(),
  CONSTRAINT profiles_user_id_unique UNIQUE (user_id)
);

-- 2. referrals table: tracks who invited whom
CREATE TABLE IF NOT EXISTS public.referrals (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  inviter_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  invitee_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ DEFAULT now(),
  CONSTRAINT referrals_invitee_unique UNIQUE (invitee_id)
);

-- 3. points_ledger table: records all points transactions
CREATE TABLE IF NOT EXISTS public.points_ledger (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  amount INTEGER NOT NULL,
  reason TEXT NOT NULL,
  reference_id UUID,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- 4. points_balance view: calculates current balance per user
CREATE OR REPLACE VIEW public.points_balance AS
SELECT 
  user_id,
  COALESCE(SUM(amount), 0)::INTEGER AS balance
FROM public.points_ledger
GROUP BY user_id;

-- 5. RLS policies
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.referrals ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.points_ledger ENABLE ROW LEVEL SECURITY;

-- profiles: users can read own profile, service role can do all
CREATE POLICY "Users can view own profile" ON public.profiles
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Service role can manage profiles" ON public.profiles
  FOR ALL USING (auth.role() = 'service_role');

-- referrals: users can view own referrals (as inviter or invitee)
CREATE POLICY "Users can view own referrals" ON public.referrals
  FOR SELECT USING (auth.uid() = inviter_id OR auth.uid() = invitee_id);

CREATE POLICY "Service role can manage referrals" ON public.referrals
  FOR ALL USING (auth.role() = 'service_role');

-- points_ledger: users can view own ledger
CREATE POLICY "Users can view own points" ON public.points_ledger
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Service role can manage points" ON public.points_ledger
  FOR ALL USING (auth.role() = 'service_role');

-- 6. Auto-create profile on user signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (user_id)
  VALUES (NEW.id)
  ON CONFLICT (user_id) DO NOTHING;
  
  -- Award signup bonus points
  INSERT INTO public.points_ledger (user_id, amount, reason)
  VALUES (NEW.id, 100, 'signup_bonus');
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- 7. apply_invite_code RPC: claim invite code and award points
CREATE OR REPLACE FUNCTION public.apply_invite_code(p_code TEXT)
RETURNS JSON AS $$
DECLARE
  v_inviter_id UUID;
  v_invitee_id UUID := auth.uid();
  v_existing_referral UUID;
  v_inviter_points INTEGER := 50;
  v_invitee_points INTEGER := 50;
BEGIN
  -- Check if user is logged in
  IF v_invitee_id IS NULL THEN
    RETURN json_build_object('success', false, 'error', 'not_authenticated');
  END IF;

  -- Check if already claimed an invite code
  SELECT id INTO v_existing_referral
  FROM public.referrals
  WHERE invitee_id = v_invitee_id;
  
  IF v_existing_referral IS NOT NULL THEN
    RETURN json_build_object('success', false, 'error', 'already_claimed');
  END IF;

  -- Find inviter by invite code
  SELECT user_id INTO v_inviter_id
  FROM public.profiles
  WHERE invite_code = upper(p_code);
  
  IF v_inviter_id IS NULL THEN
    RETURN json_build_object('success', false, 'error', 'invalid_code');
  END IF;

  -- Cannot use own invite code
  IF v_inviter_id = v_invitee_id THEN
    RETURN json_build_object('success', false, 'error', 'self_invite');
  END IF;

  -- Create referral record
  INSERT INTO public.referrals (inviter_id, invitee_id)
  VALUES (v_inviter_id, v_invitee_id);

  -- Award points to inviter
  INSERT INTO public.points_ledger (user_id, amount, reason, reference_id)
  VALUES (v_inviter_id, v_inviter_points, 'referral_bonus', v_invitee_id);

  -- Award points to invitee
  INSERT INTO public.points_ledger (user_id, amount, reason, reference_id)
  VALUES (v_invitee_id, v_invitee_points, 'invited_bonus', v_inviter_id);

  RETURN json_build_object('success', true, 'points_awarded', v_invitee_points);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant execute permission to authenticated users
GRANT EXECUTE ON FUNCTION public.apply_invite_code(TEXT) TO authenticated;
