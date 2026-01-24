-- Daily Usage Grant System Migration
-- Adds daily grant functionality for usage counts
-- Note: signup/referral rewards are already correct (20/10 points)

-- Create daily usage grant function
-- Grants 10 usage counts per day if balance <= 30
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
'Grants 10 daily usage counts if user balance <= 30. Idempotent - safe to call multiple times per day. Uses event_key for deduplication.';
