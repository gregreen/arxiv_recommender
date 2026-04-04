import { useEffect, useRef, useState } from "react";

interface NavMenuProps {
  email: string | undefined;
  onLogout: () => void;
}

export default function NavMenu({ email, onLogout }: NavMenuProps) {
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
        <span className="text-sm text-gray-500 truncate max-w-48">{email}</span>
        <button
          onClick={onLogout}
          className="text-sm text-gray-500 hover:text-red-600 transition-colors whitespace-nowrap"
        >
          Sign out
        </button>
      </div>

      {/* Mobile: hamburger */}
      <div ref={ref} className="relative ml-auto md:hidden">
        <button
          onClick={() => setOpen((v) => !v)}
          aria-label="Menu"
          className="p-1.5 rounded text-gray-600 hover:bg-gray-100 transition-colors"
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
