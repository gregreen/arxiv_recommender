import { NavLink, Outlet, useLocation } from "react-router-dom";
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
  const location = useLocation();
  // On mobile: sidebar is the landing screen; any sub-route slides it away.
  const isSubRoute = location.pathname !== "/admin" && location.pathname !== "/admin/";

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

      {/* Body: sidebar + main content, with mobile slide transition */}
      <div className="relative flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className={`
          absolute inset-0 z-10 flex flex-col bg-white transition-transform duration-300 ease-in-out
          md:relative md:w-44 md:shrink-0 md:border-r md:border-gray-200 md:translate-x-0 md:pt-4
          ${isSubRoute ? "-translate-x-full" : "translate-x-0"}
        `}>
          {NAV_ITEMS.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `px-4 py-4 md:py-2.5 text-base md:text-sm font-medium transition-colors border-b border-gray-100 md:border-none ${
                  isActive
                    ? "bg-red-50 text-red-700 md:border-r-2 md:border-red-700"
                    : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"
                }`
              }
            >
              {label}
            </NavLink>
          ))}
        </aside>

        {/* Main content area */}
        <main className={`
          absolute inset-0 flex flex-col bg-white transition-transform duration-300 ease-in-out
          md:relative md:flex-1 md:min-w-0 md:overflow-y-auto md:translate-x-0
          ${isSubRoute ? "translate-x-0" : "translate-x-full"}
        `}>
          {/* Desktop placeholder when no sub-route is selected */}
          {!isSubRoute && (
            <div className="hidden md:flex flex-1 items-center justify-center text-gray-400 text-sm">
              Select a section from the sidebar.
            </div>
          )}
          <Outlet />
        </main>
      </div>
    </div>
  );
}
