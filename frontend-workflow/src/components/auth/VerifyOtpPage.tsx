/**
 * OTP Verification page component.
 *
 * Allows users to enter the 6-digit verification code sent to their email.
 */

import { useState, useRef, useEffect } from "react";
import { useAuthStore } from "../../stores/authStore";
import { Mail, AlertCircle, Loader2, RefreshCw } from "lucide-react";

interface Props {
  email: string;
  onBack: () => void;
}

export function VerifyOtpPage({ email, onBack }: Props) {
  const [otp, setOtp] = useState(["", "", "", "", "", ""]);
  const [resendCooldown, setResendCooldown] = useState(0);
  const inputRefs = useRef<(HTMLInputElement | null)[]>([]);

  const { verifyOtp, resendOtp, loading, error, clearError } = useAuthStore();

  // Handle cooldown timer
  useEffect(() => {
    if (resendCooldown > 0) {
      const timer = setTimeout(() => setResendCooldown(resendCooldown - 1), 1000);
      return () => clearTimeout(timer);
    }
  }, [resendCooldown]);

  // Focus first input on mount
  useEffect(() => {
    inputRefs.current[0]?.focus();
  }, []);

  const handleChange = (index: number, value: string) => {
    // Only allow digits
    if (value && !/^\d$/.test(value)) return;

    const newOtp = [...otp];
    newOtp[index] = value;
    setOtp(newOtp);

    // Auto-focus next input
    if (value && index < 5) {
      inputRefs.current[index + 1]?.focus();
    }

    // Auto-submit when all digits entered
    if (value && index === 5 && newOtp.every((d) => d !== "")) {
      handleSubmit(newOtp.join(""));
    }
  };

  const handleKeyDown = (index: number, e: React.KeyboardEvent) => {
    // Handle backspace
    if (e.key === "Backspace" && !otp[index] && index > 0) {
      inputRefs.current[index - 1]?.focus();
    }
  };

  const handlePaste = (e: React.ClipboardEvent) => {
    e.preventDefault();
    const pasted = e.clipboardData.getData("text").replace(/\D/g, "").slice(0, 6);
    if (pasted.length === 6) {
      const newOtp = pasted.split("");
      setOtp(newOtp);
      handleSubmit(pasted);
    }
  };

  const handleSubmit = async (code?: string) => {
    clearError();
    const otpCode = code || otp.join("");
    if (otpCode.length !== 6) return;

    await verifyOtp(email, otpCode);
  };

  const handleResend = async () => {
    if (resendCooldown > 0) return;
    clearError();
    await resendOtp(email);
    setResendCooldown(60); // 60 second cooldown
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#0a0a1a]">
      <div className="glass-dark p-8 rounded-xl w-full max-w-md border border-white/10">
        <div className="text-center mb-6">
          <div className="w-16 h-16 bg-primary-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
            <Mail className="text-primary-400" size={32} />
          </div>
          <h2 className="text-2xl font-bold text-white mb-2">
            Check your email
          </h2>
          <p className="text-gray-400 text-sm">
            We sent a 6-digit code to <strong className="text-white">{email}</strong>
          </p>
        </div>

        {error && (
          <div className="mb-4 p-3 bg-red-500/20 border border-red-500/50 rounded-lg flex items-center gap-2 text-red-300">
            <AlertCircle size={18} />
            <span className="text-sm">{error}</span>
          </div>
        )}

        {/* OTP Input */}
        <div className="flex justify-center gap-2 mb-6">
          {otp.map((digit, index) => (
            <input
              key={index}
              ref={(el) => (inputRefs.current[index] = el)}
              type="text"
              inputMode="numeric"
              maxLength={1}
              value={digit}
              onChange={(e) => handleChange(index, e.target.value)}
              onKeyDown={(e) => handleKeyDown(index, e)}
              onPaste={handlePaste}
              disabled={loading}
              className="w-12 h-14 text-center text-2xl font-bold bg-white/5 border border-white/20 rounded-lg text-white focus:outline-none focus:border-primary-500 disabled:opacity-50 transition-colors"
            />
          ))}
        </div>

        <button
          onClick={() => handleSubmit()}
          disabled={loading || otp.some((d) => !d)}
          className="w-full py-2.5 bg-primary-500 hover:bg-primary-600 text-white font-medium rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <Loader2 size={18} className="animate-spin" />
              Verifying...
            </>
          ) : (
            "Verify"
          )}
        </button>

        {/* Resend button */}
        <div className="mt-4 text-center">
          <button
            onClick={handleResend}
            disabled={loading || resendCooldown > 0}
            className="text-sm text-gray-400 hover:text-primary-400 disabled:text-gray-600 disabled:cursor-not-allowed flex items-center justify-center gap-1 mx-auto"
          >
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
            {resendCooldown > 0
              ? `Resend code in ${resendCooldown}s`
              : "Resend code"}
          </button>
        </div>

        {/* Back button */}
        <p className="mt-6 text-center text-gray-400 text-sm">
          <button
            onClick={onBack}
            className="text-primary-400 hover:underline"
          >
            ← Back to login
          </button>
        </p>
      </div>
    </div>
  );
}
