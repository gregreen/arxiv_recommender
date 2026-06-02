import { useState } from "react";
import { useNavigate } from "react-router-dom";
import AppNav from "../components/AppNav";
import { changePassword, deleteAccount } from "../api/auth";
import { ApiError } from "../api/client";
import { useAuth } from "../AuthContext";

export default function ChangePasswordPage() {
  const { clearUser, user } = useAuth();
  const navigate = useNavigate();
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [loading, setLoading] = useState(false);

  // Danger Zone — three-layer guard
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleteAcknowledged, setDeleteAcknowledged] = useState(false);
  const [deleteEmail, setDeleteEmail] = useState("");
  const [deletePassword, setDeletePassword] = useState("");
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [deleteLoading, setDeleteLoading] = useState(false);

  const emailMatches =
    deleteEmail.trim().toLowerCase() === (user?.email ?? "").toLowerCase();
  const canSubmit =
    deleteAcknowledged && emailMatches && deletePassword.length > 0;

  function resetDeleteState() {
    setDeleteOpen(false);
    setDeleteAcknowledged(false);
    setDeleteEmail("");
    setDeletePassword("");
    setDeleteError(null);
  }

  async function handleDeleteAccount(e: React.SyntheticEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!canSubmit) return;
    setDeleteError(null);
    setDeleteLoading(true);
    try {
      await deleteAccount(deletePassword);
      clearUser();
      navigate("/login");
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        if (err.status === 400) {
          setDeleteError("Password is incorrect.");
        } else if (err.status === 403) {
          setDeleteError(err.message || "Admin accounts cannot be self-deleted.");
        } else if (err.status === 409) {
          setDeleteError(err.message || "You are the sole admin of one or more groups.");
        } else if (err.status === 429) {
          setDeleteError("Too many attempts. Please try again later.");
        } else {
          setDeleteError(err.message || "An error occurred. Please try again.");
        }
      } else {
        setDeleteError(err instanceof Error ? err.message : "An error occurred. Please try again.");
      }
      setDeleteLoading(false);
    }
  }

  async function handleSubmit(e: React.SyntheticEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);
    if (newPassword !== confirmPassword) {
      setError("New passwords do not match.");
      return;
    }
    if (newPassword.length < 8) {
      setError("New password must be at least 8 characters.");
      return;
    }
    setLoading(true);
    try {
      await changePassword(currentPassword, newPassword);
      setSuccess(true);
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        if (err.status === 400) {
          setError("Current password is incorrect.");
        } else if (err.status === 429) {
          setError("Too many attempts. Please try again later.");
        } else {
          setError(err.message || "An error occurred. Please try again.");
        }
      } else {
        setError(err instanceof Error ? err.message : "An error occurred. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col min-h-screen bg-gray-50">
      <AppNav />
      <div className="max-w-sm mx-auto mt-12 px-4 w-full">
        <div className="bg-white shadow rounded-lg p-8">
          <h1 className="text-2xl font-bold mb-6 text-gray-800">Change Password</h1>
          {success && (
            <div className="mb-4 text-sm text-green-800 bg-green-50 border border-green-200 rounded p-3">
              Password changed successfully.
            </div>
          )}
          {error && (
            <div className="mb-4 text-sm text-red-600 bg-red-50 border border-red-200 rounded p-2">
              {error}
            </div>
          )}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Current Password</label>
              <input
                type="password"
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                required
                autoFocus
                className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">New Password</label>
              <input
                type="password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                required
                className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Confirm New Password</label>
              <input
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
                className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-medium rounded px-4 py-2 text-sm transition-colors"
            >
              {loading ? "Saving…" : "Change Password"}
            </button>
          </form>
        </div>
      </div>

      <div className="max-w-sm mx-auto mt-8 px-4 w-full pb-12">
        <div className="bg-white shadow rounded-lg p-8 border border-red-200">
          <h2 className="text-lg font-semibold text-red-700 mb-1">Danger Zone</h2>
          <p className="text-sm text-gray-600 mb-4">
            Permanently delete your account and all associated data. <span className="font-bold text-red-600">This action cannot be undone!</span>
          </p>

          {/* Stage 1: reveal trigger */}
          {!deleteOpen && (
            <button
              type="button"
              onClick={() => setDeleteOpen(true)}
              className="w-full border border-red-300 text-red-600 hover:bg-red-50 font-medium rounded px-4 py-2 text-sm transition-colors"
            >
              Delete My Account
            </button>
          )}

          {/* Stages 2 & 3: confirmation form */}
          {deleteOpen && (
            <>
              {deleteError && (
                <div className="mb-4 text-sm text-red-600 bg-red-50 border border-red-200 rounded p-2">
                  {deleteError}
                </div>
              )}
              <form onSubmit={handleDeleteAccount} className="space-y-4">
                {/* Stage 2: acknowledgement checkbox */}
                <label className="flex items-start gap-2 text-sm text-gray-700 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={deleteAcknowledged}
                    onChange={(e) => setDeleteAcknowledged(e.target.checked)}
                    className="mt-0.5 accent-red-600"
                  />
                  <span>
                    I understand that deleting my account is permanent and cannot be undone.
                  </span>
                </label>

                {/* Stage 3a: type-to-confirm email */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Type your email address to confirm
                    <span className="block font-normal text-gray-400">({user?.email})</span>
                  </label>
                  <input
                    type="email"
                    value={deleteEmail}
                    onChange={(e) => setDeleteEmail(e.target.value)}
                    autoComplete="off"
                    readOnly
                    onFocus={(e) => e.currentTarget.removeAttribute("readonly")}
                    className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-red-500"
                  />
                </div>

                {/* Stage 3b: password */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Enter your password
                  </label>
                  <input
                    type="password"
                    value={deletePassword}
                    onChange={(e) => setDeletePassword(e.target.value)}
                    autoComplete="new-password"
                    readOnly
                    onFocus={(e) => e.currentTarget.removeAttribute("readonly")}
                    className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-red-500"
                  />
                </div>

                <div className="flex gap-2 pt-1">
                  <button
                    type="button"
                    onClick={resetDeleteState}
                    className="flex-1 border border-gray-300 text-gray-600 hover:bg-gray-50 font-medium rounded px-4 py-2 text-sm transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={!canSubmit || deleteLoading}
                    className="flex-1 bg-red-600 hover:bg-red-700 disabled:opacity-40 text-white font-medium rounded px-4 py-2 text-sm transition-colors"
                  >
                    {deleteLoading ? "Deleting…" : "Delete My Account"}
                  </button>
                </div>
              </form>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
