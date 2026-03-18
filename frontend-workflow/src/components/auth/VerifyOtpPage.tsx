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
  const [otpLength, setOtpLength] = useState(6);
  const [otp, setOtp] = useState(Array(6).fill(""));
  const [resendCooldown, setResendCooldown] = useState(0);
  const inputRefs = useRef<(HTMLInputElement | null)[]>([]);

  const { verifyOtp, resendOtp, loading, error, clearError } = useAuthStore();

  // Reset OTP array when length changes
  useEffect(() => {
    setOtp(Array(otpLength).fill(""));
    // Reset refs array
    inputRefs.current = inputRefs.current.slice(0, otpLength);
    // Focus first input
    setTimeout(() => inputRefs.current[0]?.focus(), 0);
  }, [otpLength]);

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
    if (value && index < otpLength - 1) {
      inputRefs.current[index + 1]?.focus();
    }

    // Auto-submit when all digits entered
    if (value && index === otpLength - 1 && newOtp.every((d) => d !== "")) {
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
    const pasted = e.clipboardData.getData("text").replace(/\D/g, "");
    
    // Auto-detect length
    if (pasted.length === 8 && otpLength !== 8) {
      setOtpLength(8);
      // We need to wait for the effect to update state and refs
      setTimeout(() => {
        const newOtp = pasted.split("");
        setOtp(newOtp);
        handleSubmit(pasted);
      }, 50);
      return;
    }
    
    const relevantPasted = pasted.slice(0, otpLength);
    if (relevantPasted.length === otpLength) {
      const newOtp = relevantPasted.split("");
      setOtp(newOtp);
      handleSubmit(relevantPasted);
    }
  };

  const handleSubmit = async (code?: string) => {
    clearError();
    const otpCode = code || otp.join("");
    if (otpCode.length !== otpLength) return;

    await verifyOtp(email, otpCode);
  };

  const handleResend = async () => {
    if (resendCooldown > 0) return;
    clearError();
    await resendOtp(email);
    setResendCooldown(60); // 60 second cooldown
  };

  return (
    <div className="portal-shell min-h-screen flex items-center justify-center p-4">
      <div className="portal-panel p-8 rounded-[28px] w-full max-w-md">
        <div className="text-center mb-6">
          <div className="w-16 h-16 bg-primary-500/10 rounded-full flex items-center justify-center mx-auto mb-4">
            <Mail className="text-primary-600" size={32} />
          </div>
          <h2 className="text-2xl font-bold font-display text-primary-900 mb-2">
            Check your email
          </h2>
          <p className="text-slate-500 text-sm">
            We sent a verification code to <strong className="text-primary-900">{email}</strong>
          </p>
        </div>

        {error && (
          <div className="mb-4 p-3 bg-red-500/20 border border-red-500/50 rounded-lg flex items-center gap-2 text-red-300">
            <AlertCircle size={18} />
            <span className="text-sm">{error}</span>
          </div>
        )}

        {/* OTP Input */}
        <div className="flex justify-center gap-2 mb-4">
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
              className="portal-input w-10 h-14 sm:w-12 text-center text-xl sm:text-2xl font-bold rounded-xl disabled:opacity-50 transition-colors"
            />
          ))}
        </div>

        {/* Length Toggle */}
        <div className="text-center mb-6">
          <button
            onClick={() => setOtpLength(otpLength === 6 ? 8 : 6)}
            className="text-xs text-slate-500 hover:text-primary-700 transition-colors"
          >
            {otpLength === 6 ? "Use 8-digit code" : "Use 6-digit code"}
          </button>
        </div>

        <button
          onClick={() => handleSubmit()}
          disabled={loading || otp.some((d) => !d)}
          className="portal-button-primary w-full py-2.5 font-medium rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
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
            className="text-sm text-slate-500 hover:text-primary-600 disabled:text-slate-300 disabled:cursor-not-allowed flex items-center justify-center gap-1 mx-auto"
          >
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
            {resendCooldown > 0
              ? `Resend code in ${resendCooldown}s`
              : "Resend code"}
          </button>
        </div>

        {/* Back button */}
        <p className="mt-6 text-center text-slate-500 text-sm">
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
