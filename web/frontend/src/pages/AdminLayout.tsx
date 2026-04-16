import { NavLink, Outlet } from "react-router-dom";
import { logout } from "../api/auth";
import { useAuth } from "../AuthContext";
import NavMenu from "../components/NavMenu";

const NAV_ITEMS = [
  { to: "users",  label: "Users" },
  { to: "tasks",  label: "Tasks" },
  { to: "papers", label: "Papers" },
  { to: "groups", label: "Groups" },
] as const;

export default function AdminLayout() {
  const { user, clearUser } = useAuth();

  async function handleLogout() {
    await logout().catch(() => {});
    clearUser();
  }

  return (
    <div className="flex flex-col h-screen overflow-x-hidden bg-gray-100">
      {/* Top navbar */}
      <nav className="flex items-center gap-4 px-4 py-2 bg-red-700 text-white shrink-0">
        <span className="font-bold text-lg tracking-tight">arXiv Recommender</span>
        <span className="text-red-200 text-sm font-semibold uppercase tracking-widest">Admin</span>
        <NavMenu email={user?.email} onLogout={handleLogout} adminMode />
      </nav>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className="w-44 shrink-0 bg-white border-r border-gray-200 flex flex-col pt-4">
          {NAV_ITEMS.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `px-4 py-2.5 text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-red-50 text-red-700 border-r-2 border-red-700"
                    : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"
                }`
              }
            >
              {label}
            </NavLink>
          ))}
        </aside>

        {/* Main content area */}
        <main className="flex-1 min-w-0 overflow-y-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
