import { useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { requestPasswordReset } from "../api/auth";
import { ApiError } from "../api/client";

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await requestPasswordReset(email);
      setSubmitted(true);
    } catch (err: unknown) {
      if (err instanceof ApiError && err.status === 429) {
        setError(err.message);
      } else {
        setError(err instanceof Error ? err.message : "An error occurred. Please try again.");
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
          <h1 className="text-2xl font-bold mb-6 text-gray-800">Forgot Password</h1>
          {submitted ? (
            <div className="text-sm text-green-800 bg-green-50 border border-green-200 rounded p-3">
              If that email address is registered, a password reset link has been sent. Please check your inbox.
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
                  <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                    autoFocus
                    className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <button
                  type="submit"
                  disabled={loading}
                  className="w-full bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-medium rounded px-4 py-2 text-sm transition-colors"
                >
                  {loading ? "Sending…" : "Send Reset Link"}
                </button>
              </form>
            </>
          )}
          <p className="mt-4 text-sm text-gray-500 text-center">
            <Link to="/login" className="text-blue-600 hover:underline">Back to Sign In</Link>
          </p>
        </div>
      </div>
    </div>
  );
}
