// Run:
//   ALIYUN_ACCESS_KEY_ID=... ALIYUN_ACCESS_KEY_SECRET=... \
//   deno run -A scripts/aliyun_dypns_test.ts --phone 17864260105 --otp 123456
// Optional env:
//   ALIYUN_REGION_ID=cn-hangzhou
//   ALIYUN_SMS_SIGN_NAME=速通互联验证码
//   ALIYUN_SMS_TEMPLATE_CODE=100001
//   ALIYUN_SMS_TEMPLATE_MIN=5
//   ALIYUN_SMS_COUNTRY_CODE=86
//   ALIYUN_SMS_ENDPOINT=https://dypnsapi.aliyuncs.com
//   ALIYUN_SMS_VERSION=2017-05-25

export {};

declare const Deno: {
  env: { get: (name: string) => string | undefined };
  args: string[];
};

function getEnv(name: string, fallback?: string): string | undefined {
  const v = Deno.env.get(name);
  return v && v.length > 0 ? v : fallback;
}

function requireEnv(name: string): string {
  const v = getEnv(name);
  if (!v) throw new Error(`Missing env: ${name}`);
  return v;
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

function buildCanonicalizedQuery(params: Record<string, string>): string {
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

function parseArgs(argv: string[]): { phone: string; otp: string } {
  const get = (k: string): string | undefined => {
    const idx = argv.indexOf(k);
    if (idx >= 0 && idx + 1 < argv.length) return argv[idx + 1];
    const prefix = `${k}=`;
    const found = argv.find((a) => a.startsWith(prefix));
    return found ? found.slice(prefix.length) : undefined;
  };

  const phone = get("--phone") ?? get("-p");
  const otp = get("--otp") ?? get("-o");
  if (!phone || !otp) {
    throw new Error("Usage: deno run -A scripts/aliyun_dypns_test.ts --phone <phone> --otp <otp>");
  }
  return { phone: normalizeChinaPhone(phone), otp };
}

async function main() {
  const { phone, otp } = parseArgs(Deno.args);

  const accessKeyId = requireEnv("ALIYUN_ACCESS_KEY_ID");
  const accessKeySecret = requireEnv("ALIYUN_ACCESS_KEY_SECRET");

  const regionId = getEnv("ALIYUN_REGION_ID", "cn-hangzhou")!;
  const endpoint = getEnv("ALIYUN_SMS_ENDPOINT", "https://dypnsapi.aliyuncs.com")!;
  const version = getEnv("ALIYUN_SMS_VERSION", "2017-05-25")!;

  const signName = getEnv("ALIYUN_SMS_SIGN_NAME", "速通互联验证码")!;
  const templateCode = getEnv("ALIYUN_SMS_TEMPLATE_CODE", "100001")!;
  const countryCode = getEnv("ALIYUN_SMS_COUNTRY_CODE", "86")!;
  const templateMin = getEnv("ALIYUN_SMS_TEMPLATE_MIN", "5")!;

  const nonceBytes = crypto.getRandomValues(new Uint8Array(16));
  const signatureNonce = toHex(nonceBytes);

  const params: Record<string, string> = {
    Action: "SendSmsVerifyCode",
    Version: version,
    Format: "JSON",
    RegionId: regionId,
    AccessKeyId: accessKeyId,
    SignatureMethod: "HMAC-SHA1",
    SignatureVersion: "1.0",
    SignatureNonce: signatureNonce,
    Timestamp: new Date().toISOString(),

    CountryCode: countryCode,
    PhoneNumber: phone,
    SignName: signName,
    TemplateCode: templateCode,
    TemplateParam: JSON.stringify({ code: otp, min: templateMin }),
  };

  const canonicalizedQuery = buildCanonicalizedQuery(params);
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

  const result = {
    ok: res.ok && (json as any)?.Code === "OK",
    status: res.status,
    headers: {
      "x-acs-request-id": res.headers.get("x-acs-request-id"),
      "x-acs-trace-id": res.headers.get("x-acs-trace-id"),
    },
    body: json,
  };

  console.log(JSON.stringify(result, null, 2));
}

await main();
