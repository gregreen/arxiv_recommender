import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import { ApiError } from "../api/client";
import { getMyGroups, type Group } from "../api/groups";
import { useAuth } from "../AuthContext";

interface GroupsState {
  groups: Group[];
  isLoading: boolean;
  refetch: () => void;
}

const GroupsContext = createContext<GroupsState>({
  groups: [],
  isLoading: true,
  refetch: () => {},
});

export function GroupsProvider({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  const [groups, setGroups] = useState<Group[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const fetchGroups = useCallback(() => {
    if (!user) {
      setGroups([]);
      setIsLoading(false);
      return;
    }
    setIsLoading(true);
    getMyGroups()
      .then(setGroups)
      .catch((err) => {
        if (!(err instanceof ApiError && err.status === 401)) {
          console.error("Failed to load groups", err);
        }
        setGroups([]);
      })
      .finally(() => setIsLoading(false));
  }, [user]);

  useEffect(() => {
    fetchGroups();
  }, [fetchGroups]);

  return (
    <GroupsContext.Provider value={{ groups, isLoading, refetch: fetchGroups }}>
      {children}
    </GroupsContext.Provider>
  );
}

export function useGroups(): GroupsState {
  return useContext(GroupsContext);
}
