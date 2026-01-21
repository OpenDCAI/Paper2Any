


SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;


COMMENT ON SCHEMA "public" IS 'standard public schema';



CREATE EXTENSION IF NOT EXISTS "pg_graphql" WITH SCHEMA "graphql";






CREATE EXTENSION IF NOT EXISTS "pg_stat_statements" WITH SCHEMA "extensions";






CREATE EXTENSION IF NOT EXISTS "pgcrypto" WITH SCHEMA "extensions";






CREATE EXTENSION IF NOT EXISTS "supabase_vault" WITH SCHEMA "vault";






CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA "extensions";






CREATE OR REPLACE FUNCTION "public"."apply_invite_code"("p_code" "text") RETURNS json
    LANGUAGE "plpgsql" SECURITY DEFINER
    AS $$
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

  -- Award points to inviter with event_key
  INSERT INTO public.points_ledger (user_id, points, reason, event_key)
  VALUES (v_inviter_id, v_inviter_points, 'referral_inviter', 'referral_inviter_' || v_inviter_id::text || '_' || v_invitee_id::text);

  -- Award points to invitee with event_key
  INSERT INTO public.points_ledger (user_id, points, reason, event_key)
  VALUES (v_invitee_id, v_invitee_points, 'referral_invitee', 'referral_invitee_' || v_invitee_id::text);

  RETURN json_build_object('success', true, 'inviter_id', v_inviter_id);
END;
$$;


ALTER FUNCTION "public"."apply_invite_code"("p_code" "text") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."deduct_points"("p_user_id" "uuid", "p_amount" integer, "p_reason" "text") RETURNS boolean
    LANGUAGE "plpgsql" SECURITY DEFINER
    AS $$
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
$$;


ALTER FUNCTION "public"."deduct_points"("p_user_id" "uuid", "p_amount" integer, "p_reason" "text") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."generate_invite_code"() RETURNS "text"
    LANGUAGE "plpgsql"
    AS $$
declare
  raw text;
begin
  -- Use extensions schema for gen_random_bytes
  raw := encode(extensions.gen_random_bytes(10), 'hex');
  return upper(substr(raw, 1, 10));
end;
$$;


ALTER FUNCTION "public"."generate_invite_code"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."handle_new_user"() RETURNS "trigger"
    LANGUAGE "plpgsql" SECURITY DEFINER
    AS $$
BEGIN
  INSERT INTO public.profiles (user_id)
  VALUES (NEW.id)
  ON CONFLICT (user_id) DO NOTHING;
  
  -- Award signup bonus points with event_key
  INSERT INTO public.points_ledger (user_id, points, reason, event_key)
  VALUES (NEW.id, 20, 'signup_bonus', 'signup_bonus_' || NEW.id::text)
  ON CONFLICT (event_key) DO NOTHING;
  
  RETURN NEW;
END;
$$;


ALTER FUNCTION "public"."handle_new_user"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."handle_new_user_create_profile"() RETURNS "trigger"
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
declare
  code text;
begin
  -- Skip anonymous users (their id is not a valid UUID or is_anonymous is true)
  IF new.is_anonymous = true THEN
    return new;
  END IF;

  -- Retry a few times in case of rare unique collision
  for i in 1..10 loop
    code := public.generate_invite_code();
    begin
      insert into public.profiles(user_id, invite_code)
      values (new.id, code);
      exit;
    exception when unique_violation then
      -- retry
    end;
  end loop;

  return new;
end;
$$;


ALTER FUNCTION "public"."handle_new_user_create_profile"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."set_updated_at"() RETURNS "trigger"
    LANGUAGE "plpgsql"
    AS $$
begin
  new.updated_at = now();
  return new;
end;
$$;


ALTER FUNCTION "public"."set_updated_at"() OWNER TO "postgres";

SET default_tablespace = '';

SET default_table_access_method = "heap";


CREATE TABLE IF NOT EXISTS "public"."points_ledger" (
    "id" bigint NOT NULL,
    "user_id" "uuid" NOT NULL,
    "points" integer NOT NULL,
    "reason" "text" NOT NULL,
    "event_key" "text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."points_ledger" OWNER TO "postgres";


CREATE OR REPLACE VIEW "public"."points_balance" WITH ("security_invoker"='true') AS
 SELECT "user_id",
    COALESCE("sum"("points"), (0)::bigint) AS "balance"
   FROM "public"."points_ledger"
  GROUP BY "user_id";


ALTER VIEW "public"."points_balance" OWNER TO "postgres";


CREATE SEQUENCE IF NOT EXISTS "public"."points_ledger_id_seq"
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE "public"."points_ledger_id_seq" OWNER TO "postgres";


ALTER SEQUENCE "public"."points_ledger_id_seq" OWNED BY "public"."points_ledger"."id";



CREATE TABLE IF NOT EXISTS "public"."profiles" (
    "user_id" "uuid" NOT NULL,
    "invite_code" "text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."profiles" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."referrals" (
    "id" bigint NOT NULL,
    "inviter_user_id" "uuid" NOT NULL,
    "invitee_user_id" "uuid" NOT NULL,
    "invite_code" "text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."referrals" OWNER TO "postgres";


CREATE SEQUENCE IF NOT EXISTS "public"."referrals_id_seq"
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE "public"."referrals_id_seq" OWNER TO "postgres";


ALTER SEQUENCE "public"."referrals_id_seq" OWNED BY "public"."referrals"."id";



CREATE TABLE IF NOT EXISTS "public"."usage_records" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_id" "uuid" NOT NULL,
    "workflow_type" "text" NOT NULL,
    "called_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."usage_records" OWNER TO "postgres";


COMMENT ON TABLE "public"."usage_records" IS 'Tracks workflow API calls per user for daily rate limiting';



CREATE TABLE IF NOT EXISTS "public"."user_files" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_id" "uuid" NOT NULL,
    "file_path" "text" NOT NULL,
    "file_name" "text" NOT NULL,
    "file_size" bigint,
    "workflow_type" "text",
    "created_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."user_files" OWNER TO "postgres";


COMMENT ON TABLE "public"."user_files" IS 'Tracks files generated by workflows and stored in Supabase Storage';



ALTER TABLE ONLY "public"."points_ledger" ALTER COLUMN "id" SET DEFAULT "nextval"('"public"."points_ledger_id_seq"'::"regclass");



ALTER TABLE ONLY "public"."referrals" ALTER COLUMN "id" SET DEFAULT "nextval"('"public"."referrals_id_seq"'::"regclass");



ALTER TABLE ONLY "public"."points_ledger"
    ADD CONSTRAINT "points_ledger_event_key_key" UNIQUE ("event_key");



ALTER TABLE ONLY "public"."points_ledger"
    ADD CONSTRAINT "points_ledger_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."profiles"
    ADD CONSTRAINT "profiles_invite_code_key" UNIQUE ("invite_code");



ALTER TABLE ONLY "public"."profiles"
    ADD CONSTRAINT "profiles_pkey" PRIMARY KEY ("user_id");



ALTER TABLE ONLY "public"."referrals"
    ADD CONSTRAINT "referrals_invitee_user_id_key" UNIQUE ("invitee_user_id");



ALTER TABLE ONLY "public"."referrals"
    ADD CONSTRAINT "referrals_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."usage_records"
    ADD CONSTRAINT "usage_records_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."user_files"
    ADD CONSTRAINT "user_files_pkey" PRIMARY KEY ("id");



CREATE INDEX "idx_usage_records_user_called" ON "public"."usage_records" USING "btree" ("user_id", "called_at");



CREATE INDEX "idx_user_files_user_id" ON "public"."user_files" USING "btree" ("user_id", "created_at" DESC);



CREATE OR REPLACE TRIGGER "profiles_set_updated_at" BEFORE UPDATE ON "public"."profiles" FOR EACH ROW EXECUTE FUNCTION "public"."set_updated_at"();



ALTER TABLE ONLY "public"."points_ledger"
    ADD CONSTRAINT "points_ledger_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."profiles"
    ADD CONSTRAINT "profiles_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."referrals"
    ADD CONSTRAINT "referrals_invitee_user_id_fkey" FOREIGN KEY ("invitee_user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."referrals"
    ADD CONSTRAINT "referrals_inviter_user_id_fkey" FOREIGN KEY ("inviter_user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."usage_records"
    ADD CONSTRAINT "usage_records_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."user_files"
    ADD CONSTRAINT "user_files_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



CREATE POLICY "Users insert own usage" ON "public"."usage_records" FOR INSERT WITH CHECK (("auth"."uid"() = "user_id"));



CREATE POLICY "Users manage own files" ON "public"."user_files" USING (("auth"."uid"() = "user_id")) WITH CHECK (("auth"."uid"() = "user_id"));



CREATE POLICY "Users read own usage" ON "public"."usage_records" FOR SELECT USING (("auth"."uid"() = "user_id"));



ALTER TABLE "public"."points_ledger" ENABLE ROW LEVEL SECURITY;


CREATE POLICY "points_select_own" ON "public"."points_ledger" FOR SELECT USING (("auth"."uid"() = "user_id"));



ALTER TABLE "public"."profiles" ENABLE ROW LEVEL SECURITY;


CREATE POLICY "profiles_select_own" ON "public"."profiles" FOR SELECT USING (("auth"."uid"() = "user_id"));



CREATE POLICY "profiles_update_own" ON "public"."profiles" FOR UPDATE USING (("auth"."uid"() = "user_id"));



ALTER TABLE "public"."referrals" ENABLE ROW LEVEL SECURITY;


CREATE POLICY "referrals_select_related" ON "public"."referrals" FOR SELECT USING ((("auth"."uid"() = "inviter_user_id") OR ("auth"."uid"() = "invitee_user_id")));



ALTER TABLE "public"."usage_records" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."user_files" ENABLE ROW LEVEL SECURITY;




ALTER PUBLICATION "supabase_realtime" OWNER TO "postgres";


GRANT USAGE ON SCHEMA "public" TO "postgres";
GRANT USAGE ON SCHEMA "public" TO "anon";
GRANT USAGE ON SCHEMA "public" TO "authenticated";
GRANT USAGE ON SCHEMA "public" TO "service_role";

























































































































































GRANT ALL ON FUNCTION "public"."apply_invite_code"("p_code" "text") TO "anon";
GRANT ALL ON FUNCTION "public"."apply_invite_code"("p_code" "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."apply_invite_code"("p_code" "text") TO "service_role";



GRANT ALL ON FUNCTION "public"."deduct_points"("p_user_id" "uuid", "p_amount" integer, "p_reason" "text") TO "anon";
GRANT ALL ON FUNCTION "public"."deduct_points"("p_user_id" "uuid", "p_amount" integer, "p_reason" "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."deduct_points"("p_user_id" "uuid", "p_amount" integer, "p_reason" "text") TO "service_role";



GRANT ALL ON FUNCTION "public"."generate_invite_code"() TO "anon";
GRANT ALL ON FUNCTION "public"."generate_invite_code"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."generate_invite_code"() TO "service_role";



GRANT ALL ON FUNCTION "public"."handle_new_user"() TO "anon";
GRANT ALL ON FUNCTION "public"."handle_new_user"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."handle_new_user"() TO "service_role";



GRANT ALL ON FUNCTION "public"."handle_new_user_create_profile"() TO "anon";
GRANT ALL ON FUNCTION "public"."handle_new_user_create_profile"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."handle_new_user_create_profile"() TO "service_role";



GRANT ALL ON FUNCTION "public"."set_updated_at"() TO "anon";
GRANT ALL ON FUNCTION "public"."set_updated_at"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."set_updated_at"() TO "service_role";


















GRANT ALL ON TABLE "public"."points_ledger" TO "anon";
GRANT ALL ON TABLE "public"."points_ledger" TO "authenticated";
GRANT ALL ON TABLE "public"."points_ledger" TO "service_role";



GRANT ALL ON TABLE "public"."points_balance" TO "anon";
GRANT ALL ON TABLE "public"."points_balance" TO "authenticated";
GRANT ALL ON TABLE "public"."points_balance" TO "service_role";



GRANT ALL ON SEQUENCE "public"."points_ledger_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."points_ledger_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."points_ledger_id_seq" TO "service_role";



GRANT ALL ON TABLE "public"."profiles" TO "anon";
GRANT ALL ON TABLE "public"."profiles" TO "authenticated";
GRANT ALL ON TABLE "public"."profiles" TO "service_role";



GRANT ALL ON TABLE "public"."referrals" TO "anon";
GRANT ALL ON TABLE "public"."referrals" TO "authenticated";
GRANT ALL ON TABLE "public"."referrals" TO "service_role";



GRANT ALL ON SEQUENCE "public"."referrals_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."referrals_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."referrals_id_seq" TO "service_role";



GRANT ALL ON TABLE "public"."usage_records" TO "anon";
GRANT ALL ON TABLE "public"."usage_records" TO "authenticated";
GRANT ALL ON TABLE "public"."usage_records" TO "service_role";



GRANT ALL ON TABLE "public"."user_files" TO "anon";
GRANT ALL ON TABLE "public"."user_files" TO "authenticated";
GRANT ALL ON TABLE "public"."user_files" TO "service_role";









ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES TO "service_role";






ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS TO "service_role";






ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES TO "service_role";































