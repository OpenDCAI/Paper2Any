/**
 * Browser fingerprint service for anonymous user identification.
 *
 * Generates a stable fingerprint based on browser characteristics.
 * Used to track quota for non-logged-in users.
 */

const FINGERPRINT_KEY = 'df_fingerprint';

/**
 * Generate a hash from a string using a simple but fast algorithm.
 */
function simpleHash(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  // Convert to hex and ensure positive
  return (hash >>> 0).toString(16).padStart(8, '0');
}

/**
 * Generate a canvas fingerprint.
 */
function getCanvasFingerprint(): string {
  try {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return '';

    // Draw some text with specific styling
    canvas.width = 200;
    canvas.height = 50;
    ctx.textBaseline = 'top';
    ctx.font = '14px Arial';
    ctx.fillStyle = '#f60';
    ctx.fillRect(0, 0, 100, 50);
    ctx.fillStyle = '#069';
    ctx.fillText('DataFlow Agent', 2, 15);
    ctx.fillStyle = 'rgba(102, 204, 0, 0.7)';
    ctx.fillText('Fingerprint', 4, 30);

    return canvas.toDataURL();
  } catch {
    return '';
  }
}

/**
 * Collect browser characteristics for fingerprinting.
 */
function collectBrowserData(): string {
  const data: string[] = [];

  // User agent
  data.push(navigator.userAgent);

  // Screen properties
  data.push(`${screen.width}x${screen.height}x${screen.colorDepth}`);

  // Timezone
  data.push(Intl.DateTimeFormat().resolvedOptions().timeZone);

  // Language
  data.push(navigator.language);

  // Platform
  data.push(navigator.platform);

  // Hardware concurrency (CPU cores)
  data.push(String(navigator.hardwareConcurrency || 0));

  // Device memory (if available)
  data.push(String((navigator as any).deviceMemory || 0));

  // Canvas fingerprint
  data.push(getCanvasFingerprint());

  // WebGL renderer (if available)
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    if (gl) {
      const debugInfo = (gl as WebGLRenderingContext).getExtension('WEBGL_debug_renderer_info');
      if (debugInfo) {
        data.push((gl as WebGLRenderingContext).getParameter(debugInfo.UNMASKED_RENDERER_WEBGL));
      }
    }
  } catch {
    // Ignore
  }

  return data.join('|');
}

/**
 * Generate or retrieve a stable browser fingerprint.
 *
 * The fingerprint is cached in localStorage for consistency.
 * Falls back to generating a new one if not found.
 *
 * @returns A hex string fingerprint (16 characters)
 */
export function getFingerprint(): string {
  // Try to get from localStorage first
  const cached = localStorage.getItem(FINGERPRINT_KEY);
  if (cached && cached.length === 16) {
    return cached;
  }

  // Generate new fingerprint
  const browserData = collectBrowserData();
  const hash1 = simpleHash(browserData);
  const hash2 = simpleHash(browserData + hash1); // Double hash for more uniqueness
  const fingerprint = hash1 + hash2;

  // Cache it
  localStorage.setItem(FINGERPRINT_KEY, fingerprint);

  return fingerprint;
}

/**
 * Get a user identifier for quota tracking.
 *
 * For logged-in users: returns Supabase user ID
 * For anonymous users: returns browser fingerprint with 'anon_' prefix
 *
 * @param userId - Supabase user ID if logged in, null otherwise
 * @returns User identifier string
 */
export function getUserIdentifier(userId: string | null): string {
  if (userId) {
    return userId;
  }
  return `anon_${getFingerprint()}`;
}
