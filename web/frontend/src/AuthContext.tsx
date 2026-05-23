import {
  createContext,
  useContext,
  useState,
  useEffect,
  type ReactNode,
} from "react";
import { ApiError } from "./api/client";
import { apiFetch } from "./api/client";

interface AuthUser {
  userId: number;
  email: string;
  isAdmin: boolean;
  tutorialShown: boolean;
}

interface AuthState {
  user: AuthUser | null;
  isLoading: boolean;
  setUser: (user: AuthUser) => void;
  clearUser: () => void;
  setTutorialShown: () => Promise<void>;
}

const AuthContext = createContext<AuthState>({
  user: null,
  isLoading: true,
  setUser: () => {},
  clearUser: () => {},
  setTutorialShown: async () => {},
});

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUserState] = useState<AuthUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check if we have a valid session by fetching the current user.
    apiFetch<{ user_id: number; email: string; is_admin: boolean; tutorial_shown: boolean }>("/api/auth/me")
      .then((data) => {
        setUserState({ userId: data.user_id, email: data.email, isAdmin: data.is_admin, tutorialShown: data.tutorial_shown });
        localStorage.setItem(
          "auth_user",
          JSON.stringify({ userId: data.user_id, email: data.email, isAdmin: data.is_admin, tutorialShown: data.tutorial_shown })
        );
      })
      .catch((err) => {
        if (err instanceof ApiError && err.status === 401) {
          localStorage.removeItem("auth_user");
          setUserState(null);
        }
      })
      .finally(() => setIsLoading(false));
  }, []);

  const setUser = (u: AuthUser) => {
    setUserState(u);
    localStorage.setItem("auth_user", JSON.stringify(u));
  };

  const clearUser = () => {
    setUserState(null);
    localStorage.removeItem("auth_user");
  };

  const setTutorialShown = async () => {
    await apiFetch("/api/auth/me", { method: "PATCH", body: JSON.stringify({ tutorial_shown: true }) });
    setUserState((prev) => prev ? { ...prev, tutorialShown: true } : prev);
  };

  return (
    <AuthContext.Provider value={{ user, isLoading, setUser, clearUser, setTutorialShown }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}

