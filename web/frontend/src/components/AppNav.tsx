import { Link, useNavigate } from "react-router-dom";
import { logout } from "../api/auth";
import { useAuth } from "../AuthContext";
import NavMenu from "./NavMenu";

/**
 * Shared top navigation bar used across all pages.
 *
 * Desktop: arXiv Recommender | Library | About  ···  email▾
 * Mobile:  arXiv Recommender | Library           ···  ☰
 *
 * When the user is not authenticated (e.g. the About page for a visitor),
 * Library is hidden and a "Sign in / register" link is shown instead of NavMenu.
 */
export default function AppNav() {
  const { user, clearUser } = useAuth();
  const navigate = useNavigate();

  async function handleLogout() {
    await logout().catch(() => {});
    clearUser();
    navigate("/login");
  }

  return (
    <nav
      className="flex items-center gap-4 px-4 py-2 border-b border-blue-200 shrink-0"
      style={{ background: "linear-gradient(42deg, #ebf5ff, #91caff)" }}
    >
      <Link to="/" className="font-bold text-blue-700 text-lg">arXiv Recommender</Link>
      {user ? (
        <>
          <Link to="/library" className="text-sm text-gray-600 hover:text-gray-900">Library</Link>
          <Link to="/about" className="hidden md:inline text-sm text-gray-600 hover:text-gray-900">About</Link>
          <NavMenu email={user.email} onLogout={handleLogout} />
        </>
      ) : (
        <Link to="/login" className="text-sm text-gray-600 hover:text-gray-900 ml-auto">
          Sign in / register
        </Link>
      )}
    </nav>
  );
}
