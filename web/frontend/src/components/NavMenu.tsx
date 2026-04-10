import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

interface NavMenuProps {
  email: string | undefined;
  onLogout: () => void;
  adminMode?: boolean;
}

export default function NavMenu({ email, onLogout, adminMode = false }: NavMenuProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [open]);

  return (
    <>
      {/* Desktop: email + sign out inline */}
      <div className="hidden md:flex items-center gap-3 ml-auto">
        <span className={`text-sm truncate max-w-48 ${adminMode ? "text-red-200" : "text-gray-900"}`}>{email}</span>
        <button
          onClick={onLogout}
          className={`text-sm transition-colors whitespace-nowrap ${adminMode ? "text-red-200 hover:text-white" : "text-gray-900 hover:text-red-600"}`}
        >
          Sign out
        </button>
      </div>

      {/* Mobile: hamburger */}
      <div ref={ref} className="relative ml-auto md:hidden">
        <button
          onClick={() => setOpen((v) => !v)}
          aria-label="Menu"
          className={`p-1.5 rounded transition-colors ${adminMode ? "text-red-200 hover:bg-red-600" : "text-gray-800 hover:bg-blue-100"}`}
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
        {open && (
          <div className="absolute right-0 top-full mt-1 w-56 bg-white border border-gray-200 rounded-lg shadow-lg z-50 py-2">
            <div className="px-4 py-2 text-sm text-gray-500 truncate border-b border-gray-100 mb-1">
              {email}
            </div>
            <Link
              to="/about"
              onClick={() => setOpen(false)}
              className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
            >
              About
            </Link>
            <button
              onClick={() => { setOpen(false); onLogout(); }}
              className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-red-50 hover:text-red-600 transition-colors"
            >
              Sign out
            </button>
          </div>
        )}
      </div>
    </>
  );
}
