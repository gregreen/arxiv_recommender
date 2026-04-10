import { useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { register } from "../api/auth";
import { ApiError } from "../api/client";

export default function RegisterPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const data = await register(email, password);
      setSuccessMessage(data.message);
    } catch (err: unknown) {
      if (err instanceof ApiError && err.status === 409) {
        setError("An account with this email already exists.");
      } else {
        setError(err instanceof Error ? err.message : "Registration failed");
      }
    } finally {
      setLoading(false);
    }
  }

  if (successMessage) {
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
        <div className="bg-white shadow rounded-lg p-8 w-full max-w-sm text-center">
          <h1 className="text-2xl font-bold mb-4 text-gray-800">Request Received</h1>
          <p className="text-gray-600 mb-4">{successMessage}</p>
          <Link to="/login" className="text-blue-600 hover:underline text-sm">
            Back to Sign In
          </Link>
        </div>
        </div>
      </div>
    );
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
        <h1 className="text-2xl font-bold mb-6 text-gray-800">Create Account</h1>
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
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={8}
              className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-medium rounded px-4 py-2 text-sm transition-colors"
          >
            {loading ? "Registering…" : "Register"}
          </button>
        </form>
        <p className="mt-4 text-sm text-gray-500 text-center">
          Already have an account?{" "}
          <Link to="/login" className="text-blue-600 hover:underline">
            Sign In
          </Link>
        </p>
      </div>
      </div>
    </div>
  );
}
