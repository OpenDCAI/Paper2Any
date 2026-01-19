declare const Deno: {
  env: { get: (name: string) => string | undefined };
  serve: (handler: (req: Request) => Response | Promise<Response>) => void;
};

import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import { Webhook } from "https://esm.sh/standardwebhooks@1.0.0";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json",
      "access-control-allow-origin": "*",
    },
  });
}

function getRequiredEnv(name: string): string {
  const valueRaw = Deno.env.get(name);
  const value = valueRaw?.trim();
  if (!value) {
    throw new Error(`Missing required env: ${name}`);
  }
  return value;
}

function decodeBase64Utf8(input: string): string {
  const bin = atob(input);
  const bytes = Uint8Array.from(bin, (c) => c.charCodeAt(0));
  return new TextDecoder().decode(bytes);
}

function resolveAliyunSmsSignName(): string {
  const fallback = "速通互联验证码";

  const b64 = Deno.env.get("ALIYUN_SMS_SIGN_NAME_B64")?.trim();
  if (b64) {
    try {
      const decoded = decodeBase64Utf8(b64).trim();
      if (decoded && !decoded.includes("\uFFFD")) return decoded;
    } catch {
      // ignore
    }
  }

  const plain = Deno.env.get("ALIYUN_SMS_SIGN_NAME")?.trim();
  if (plain && !plain.includes("\uFFFD")) return plain;

  return fallback;
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
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));

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
  const signName = resolveAliyunSmsSignName();
  const templateCode = getRequiredEnv("ALIYUN_SMS_TEMPLATE_CODE");

  const regionId = Deno.env.get("ALIYUN_REGION_ID") ?? "cn-hangzhou";

  const action = Deno.env.get("ALIYUN_SMS_ACTION") ?? "SendSmsVerifyCode";
  const version = Deno.env.get("ALIYUN_SMS_VERSION") ?? "2017-05-25";
  const countryCode = Deno.env.get("ALIYUN_SMS_COUNTRY_CODE") ?? "86";
  const templateMin = Deno.env.get("ALIYUN_SMS_TEMPLATE_MIN") ?? "5";

  if (
    (action === "SendSmsVerifyCode" || action === "CheckSmsVerifyCode") &&
    !Deno.env.get("ALIYUN_SMS_VERSION")
  ) {
    console.warn(
      `[send-sms] ALIYUN_SMS_VERSION not set; using default '${version}'. If Aliyun returns INVALID_PARAMETERS, set the correct Version from the API's '公共请求参数'.`,
    );
  }

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
    RegionId: regionId,
    AccessKeyId: accessKeyId,
    SignatureMethod: "HMAC-SHA1",
    SignatureVersion: "1.0",
    SignatureNonce: signatureNonce,
    Timestamp: new Date().toISOString(),
  };

  // Business parameters differ by product
  if (action === "SendSmsVerifyCode") {
    // As per iphone.md, this API accepts a user-provided code, but Aliyun won't verify it.
    // That's OK because Supabase generates/verifies OTP.
    params.CountryCode = countryCode;
    params.PhoneNumber = args.phoneNumber;
    params.SignName = signName;
    params.TemplateCode = templateCode;
    params.TemplateParam = JSON.stringify({ code: args.otp, min: templateMin });
  } else {
    // Default to classic SMS API
    params.PhoneNumbers = args.phoneNumber;
    params.SignName = signName;
    params.TemplateCode = templateCode;
    params.TemplateParam = JSON.stringify({ code: args.otp, min: templateMin });
  }

  const canonicalizedQuery = buildAliyunCanonicalizedQuery(params);
  const stringToSign = `GET&%2F&${percentEncode(canonicalizedQuery)}`;
  const signature = await hmacSha1Base64(`${accessKeySecret}&`, stringToSign);

  const url = `${endpoint}/?${canonicalizedQuery}&Signature=${percentEncode(signature)}`;

  const res = await fetch(url, { method: "GET" });
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
}

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS" || req.method === "GET" || req.method === "HEAD") {
    return new Response("", {
      status: 200,
      headers: {
        "content-type": "application/json",
        "access-control-allow-origin": "*",
        "access-control-allow-methods": "POST, OPTIONS",
        "access-control-allow-headers": "content-type, authorization",
      },
    });
  }

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
      return jsonResponse({}, 400);
    }

    const result = await sendAliyunSms({ phoneNumber: normalizeChinaPhone(phone), otp });

    if (!result.ok) {
      console.error("[send-sms] Aliyun SMS failed", result.raw);
      return jsonResponse({}, 500);
    }

    return jsonResponse({}, 200);
  } catch (err) {
    console.error("[send-sms] Hook failed", err);
    return jsonResponse({}, 500);
  }
});
