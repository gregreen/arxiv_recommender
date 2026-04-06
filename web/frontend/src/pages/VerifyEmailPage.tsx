import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { verifyEmail } from "../api/auth";
import { ApiError } from "../api/client";

export default function VerifyEmailPage() {
  const [searchParams] = useSearchParams();
  const token = searchParams.get("token") ?? "";

  const [status, setStatus] = useState<"loading" | "success" | "error">("loading");
  const [message, setMessage] = useState("");

  useEffect(() => {
    if (!token) {
      setStatus("error");
      setMessage("No verification token found in the URL.");
      return;
    }
    verifyEmail(token)
      .then((data) => {
        setMessage(data.message);
        setStatus("success");
      })
      .catch((err: unknown) => {
        setMessage(
          err instanceof ApiError ? err.message : "Verification failed. Please try again."
        );
        setStatus("error");
      });
  }, [token]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="bg-white shadow rounded-lg p-8 w-full max-w-sm text-center">
        {status === "loading" && (
          <p className="text-gray-500">Verifying your email…</p>
        )}
        {status === "success" && (
          <>
            <h1 className="text-2xl font-bold mb-4 text-gray-800">Email Verified</h1>
            <p className="text-gray-600 mb-4">{message}</p>
            <Link to="/login" className="text-blue-600 hover:underline text-sm">
              Sign In
            </Link>
          </>
        )}
        {status === "error" && (
          <>
            <h1 className="text-2xl font-bold mb-4 text-gray-800">Verification Failed</h1>
            <p className="text-gray-600 mb-4">{message}</p>
            <Link to="/login" className="text-blue-600 hover:underline text-sm">
              Back to Sign In
            </Link>
          </>
        )}
      </div>
    </div>
  );
}
