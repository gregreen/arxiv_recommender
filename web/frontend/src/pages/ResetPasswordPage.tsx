import { useState, type FormEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { resetPassword } from "../api/auth";
import { ApiError } from "../api/client";

export default function ResetPasswordPage() {
  const [searchParams] = useSearchParams();
  const token = searchParams.get("token") ?? "";
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);

  if (!token) {
    return (
      <div className="flex flex-col min-h-screen bg-gray-50">
        <nav
          className="flex items-center gap-4 px-4 py-2 border-b border-blue-200 shrink-0"
          style={{ background: "linear-gradient(42deg, #ebf5ff, #91caff)" }}
        >
          <Link to="/" className="font-bold text-blue-700 text-lg">arXiv Recommender</Link>
        </nav>
        <div className="flex-1 flex items-center justify-center">
          <div className="bg-white shadow rounded-lg p-8 w-full max-w-sm">
            <p className="text-sm text-red-600">Invalid reset link. Please request a new one.</p>
            <p className="mt-4 text-sm text-center">
              <Link to="/forgot-password" className="text-blue-600 hover:underline">Request password reset</Link>
            </p>
          </div>
        </div>
      </div>
    );
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    if (password !== confirm) {
      setError("Passwords do not match.");
      return;
    }
    if (password.length < 8) {
      setError("Password must be at least 8 characters.");
      return;
    }
    setLoading(true);
    try {
      await resetPassword(token, password);
      setSuccess(true);
    } catch (err: unknown) {
      if (err instanceof ApiError && err.status === 429) {
        setError(err.message);
      } else {
        setError("Invalid or expired reset link. Please request a new one.");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col min-h-screen bg-gray-50">
      <nav
        className="flex items-center gap-4 px-4 py-2 border-b border-blue-200 shrink-0"
        style={{ background: "linear-gradient(42deg, #ebf5ff, #91caff)" }}
      >
        <Link to="/" className="font-bold text-blue-700 text-lg">arXiv Recommender</Link>
        <Link to="/about" className="text-sm text-gray-600 hover:text-gray-900">About</Link>
      </nav>
      <div className="flex-1 flex items-center justify-center">
        <div className="bg-white shadow rounded-lg p-8 w-full max-w-sm">
          <h1 className="text-2xl font-bold mb-6 text-gray-800">Reset Password</h1>
          {success ? (
            <div className="text-sm text-green-800 bg-green-50 border border-green-200 rounded p-3">
              Your password has been reset.{" "}
              <Link to="/login" className="text-blue-600 hover:underline">Sign in</Link>
            </div>
          ) : (
            <>
              {error && (
                <div className="mb-4 text-sm text-red-600 bg-red-50 border border-red-200 rounded p-2">
                  {error}
                </div>
              )}
              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">New Password</label>
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    autoFocus
                    className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Confirm Password</label>
                  <input
                    type="password"
                    value={confirm}
                    onChange={(e) => setConfirm(e.target.value)}
                    required
                    className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <button
                  type="submit"
                  disabled={loading}
                  className="w-full bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-medium rounded px-4 py-2 text-sm transition-colors"
                >
                  {loading ? "Resetting…" : "Reset Password"}
                </button>
              </form>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
