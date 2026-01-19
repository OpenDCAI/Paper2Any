import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import { Webhook } from "https://esm.sh/standardwebhooks@1.0.0";

function getRequiredEnv(name: string): string {
  const value = Deno.env.get(name);
  if (!value) {
    throw new Error(`Missing required env: ${name}`);
  }
  return value;
}

function toHex(bytes: Uint8Array): string {
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function percentEncode(input: string): string {
  return encodeURIComponent(input)
    .replace(/\!/g, "%21")
    .replace(/\*/g, "%2A")
    .replace(/\(/g, "%28")
    .replace(/\)/g, "%29")
    .replace(/'/g, "%27")
    .replace(/%7E/g, "~");
}

async function hmacSha1Base64(key: string, message: string): Promise<string> {
  const enc = new TextEncoder();
  const cryptoKey = await crypto.subtle.importKey(
    "raw",
    enc.encode(key),
    { name: "HMAC", hash: "SHA-1" },
    false,
    ["sign"],
  );
  const signature = await crypto.subtle.sign("HMAC", cryptoKey, enc.encode(message));
  const bytes = new Uint8Array(signature);
  let binary = "";
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary);
}

function buildAliyunCanonicalizedQuery(params: Record<string, string>): string {
  const entries = Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== null)
    .sort(([a], [b]) => a.localeCompare(b));

  return entries
    .map(([k, v]) => `${percentEncode(k)}=${percentEncode(v)}`)
    .join("&");
}

function normalizeChinaPhone(input: string): string {
  let s = input.trim();
  if (s.startsWith("+86")) s = s.slice(3);
  if (s.startsWith("86")) s = s.slice(2);
  s = s.replace(/\D/g, "");
  return s;
}

async function sendAliyunSms(args: {
  phoneNumber: string;
  otp: string;
}): Promise<{ ok: boolean; raw: unknown }> {
  const accessKeyId = getRequiredEnv("ALIYUN_ACCESS_KEY_ID");
  const accessKeySecret = getRequiredEnv("ALIYUN_ACCESS_KEY_SECRET");
  const signName = getRequiredEnv("ALIYUN_SMS_SIGN_NAME");
  const templateCode = getRequiredEnv("ALIYUN_SMS_TEMPLATE_CODE");

  const action = Deno.env.get("ALIYUN_SMS_ACTION") ?? "SendSmsVerifyCode";
  const version = Deno.env.get("ALIYUN_SMS_VERSION") ?? "2017-05-25";
  const countryCode = Deno.env.get("ALIYUN_SMS_COUNTRY_CODE") ?? "86";
  const templateMin = Deno.env.get("ALIYUN_SMS_TEMPLATE_MIN") ?? "5";

  const endpoint = Deno.env.get("ALIYUN_SMS_ENDPOINT") ??
    (action === "SendSmsVerifyCode"
      ? "https://dypnsapi.aliyuncs.com"
      : "https://dysmsapi.aliyuncs.com");

  const nonceBytes = crypto.getRandomValues(new Uint8Array(16));
  const signatureNonce = toHex(nonceBytes);

  const params: Record<string, string> = {
    Action: action,
    Version: version,
    Format: "JSON",
    AccessKeyId: accessKeyId,
    SignatureMethod: "HMAC-SHA1",
    SignatureVersion: "1.0",
    SignatureNonce: signatureNonce,
    Timestamp: new Date().toISOString(),
  };

  if (action === "SendSmsVerifyCode") {
    params.CountryCode = countryCode;
    params.PhoneNumber = args.phoneNumber;
    params.SignName = signName;
    params.TemplateCode = templateCode;
    params.TemplateParam = JSON.stringify({ code: args.otp, min: templateMin });
  } else {
    params.PhoneNumbers = args.phoneNumber;
    params.SignName = signName;
    params.TemplateCode = templateCode;
    params.TemplateParam = JSON.stringify({ code: args.otp, min: templateMin });
  }

  const canonicalizedQuery = buildAliyunCanonicalizedQuery(params);
  const stringToSign = `GET&%2F&${percentEncode(canonicalizedQuery)}`;
  const signature = await hmacSha1Base64(`${accessKeySecret}&`, stringToSign);

  const url = `${endpoint}/?${canonicalizedQuery}&Signature=${percentEncode(signature)}`;

  // Add timeout control (3 seconds)
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 3000);

  try {
    const res = await fetch(url, { 
      method: "GET",
      signal: controller.signal 
    });
    clearTimeout(timeoutId);

    const text = await res.text();

    let json: unknown = text;
    try {
      json = JSON.parse(text);
    } catch {
      // ignore
    }

    if (!res.ok) {
      return { ok: false, raw: json };
    }

    const code = (json as { Code?: string } | null)?.Code;
    if (code && code !== "OK") {
      return { ok: false, raw: json };
    }

    return { ok: true, raw: json };
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof Error && error.name === 'AbortError') {
      return { ok: false, raw: { error: "Request timeout after 3 seconds" } };
    }
    throw error;
  }
}

Deno.serve(async (req: Request) => {
  try {
    const hookSecretRaw = getRequiredEnv("SEND_SMS_HOOK_SECRETS");
    const hookSecret = hookSecretRaw.replace("v1,whsec_", "");

    const payload = await req.text();
    const headers = Object.fromEntries(req.headers);

    const wh = new Webhook(hookSecret);
    const data = wh.verify(payload, headers) as {
      user?: { phone?: string | null };
      sms?: { otp?: string | null };
    };

    const phone = data?.user?.phone;
    const otp = data?.sms?.otp;

    if (!phone || !otp) {
      return new Response("", { status: 400 });
    }

    const result = await sendAliyunSms({
      phoneNumber: normalizeChinaPhone(phone),
      otp,
    });

    if (!result.ok) {
      console.error("[send-sms-optimized] Aliyun SMS failed", result.raw);
      return new Response("", { status: 500 });
    }

    return new Response("", { status: 200 });
  } catch (err) {
    console.error("[send-sms-optimized] Hook failed", err);
    return new Response("", { status: 500 });
  }
});
