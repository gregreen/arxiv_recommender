import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useGroups } from "../contexts/GroupsContext";

interface NavMenuProps {
  email: string | undefined;
  onLogout: () => void;
  adminMode?: boolean;
}

export default function NavMenu({ email, onLogout, adminMode = false }: NavMenuProps) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [desktopOpen, setDesktopOpen] = useState(false);
  const mobileRef = useRef<HTMLDivElement>(null);
  const desktopRef = useRef<HTMLDivElement>(null);
  const { groups } = useGroups();

  useEffect(() => {
    if (!mobileOpen && !desktopOpen) return;
    function handleClick(e: MouseEvent) {
      if (mobileRef.current && !mobileRef.current.contains(e.target as Node)) {
        setMobileOpen(false);
      }
      if (desktopRef.current && !desktopRef.current.contains(e.target as Node)) {
        setDesktopOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [mobileOpen, desktopOpen]);

  const groupsHref = groups.length === 0 ? "/groups/new" : "/groups";

  return (
    <>
      {/* Desktop: email ▾ dropdown */}
      <div ref={desktopRef} className="hidden md:flex items-center ml-auto relative">
        <button
          onClick={() => setDesktopOpen((v) => !v)}
          className={`flex items-center gap-1 text-sm cursor-pointer transition-colors ${adminMode ? "text-red-200 hover:text-white" : "text-gray-700 hover:text-gray-900"}`}
        >
          <span className="truncate max-w-48">{email}</span>
          <svg className="w-3 h-3 shrink-0" fill="none" stroke="currentColor" strokeWidth={2.5} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        {desktopOpen && (
          <div className="absolute right-0 top-full mt-1 w-44 bg-white border border-gray-200 rounded-lg shadow-lg z-50 py-1">
            <Link
              to={groupsHref}
              onClick={() => setDesktopOpen(false)}
              className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
            >
              Groups
            </Link>
            <Link
              to="/account"
              onClick={() => setDesktopOpen(false)}
              className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
            >
              Security
            </Link>
            <div className="border-t border-gray-100 my-1" />
            <button
              onClick={() => { setDesktopOpen(false); onLogout(); }}
              className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-red-50 hover:text-red-600 transition-colors"
            >
              Sign out
            </button>
          </div>
        )}
      </div>

      {/* Mobile: hamburger */}
      <div ref={mobileRef} className="relative ml-auto md:hidden">
        <button
          onClick={() => setMobileOpen((v) => !v)}
          aria-label="Menu"
          className={`p-1.5 rounded transition-colors ${adminMode ? "text-red-200 hover:bg-red-600" : "text-gray-800 hover:bg-blue-100"}`}
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
        {mobileOpen && (
          <div className="absolute right-0 top-full mt-1 w-56 bg-white border border-gray-200 rounded-lg shadow-lg z-50 py-2">
            <div className="px-4 py-2 text-sm text-gray-500 truncate border-b border-gray-100 mb-1">
              {email}
            </div>
            <Link
              to="/about"
              onClick={() => setMobileOpen(false)}
              className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
            >
              About
            </Link>
            <Link
              to={groupsHref}
              onClick={() => setMobileOpen(false)}
              className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
            >
              Groups
            </Link>
            <Link
              to="/account"
              onClick={() => setMobileOpen(false)}
              className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
            >
              Security
            </Link>
            <button
              onClick={() => { setMobileOpen(false); onLogout(); }}
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
