"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { fetchServerStatus } from "@/lib/api";

/**
 * The one state every route names.
 *
 * `checking` until /health answers. `unreachable` when it never does, which is
 * not the same as a live server. The two live states come from the two flags
 * /health reports: an operator who published a read-only demo sets demo mode,
 * and the app turns the same flag on for a machine with no model key. Reading
 * demo mode alone calls that machine a hosted demo.
 */
export type ServerState = "checking" | "unreachable" | "hosted-read-only" | "live-local";

export interface ServerStateValue {
  state: ServerState;
  // True when /chat, arena matches, graph builds and the tools answer 503.
  writesOff: boolean;
  refresh: () => void;
}

const FALLBACK: ServerStateValue = {
  state: "checking",
  writesOff: false,
  refresh: () => {},
};

const ServerStateContext = createContext<ServerStateValue>(FALLBACK);

export function useServerState(): ServerStateValue {
  return useContext(ServerStateContext);
}

export default function ServerStateProvider({ children }: { children: React.ReactNode }) {
  const [value, setValue] = useState<Omit<ServerStateValue, "refresh">>({
    state: "checking",
    writesOff: false,
  });
  const [attempt, setAttempt] = useState(0);

  const refresh = useCallback(() => {
    setValue({ state: "checking", writesOff: false });
    setAttempt((n) => n + 1);
  }, []);

  useEffect(() => {
    let active = true;
    fetchServerStatus().then((status) => {
      if (!active) return;
      if (!status) {
        setValue({ state: "unreachable", writesOff: false });
        return;
      }
      setValue({
        // Demo mode the app turned on for itself says nothing about hosting.
        // It says this machine has no model key, so live runs are off.
        state: status.demoMode && !status.demoModeAuto ? "hosted-read-only" : "live-local",
        writesOff: status.demoMode,
      });
    });
    return () => {
      active = false;
    };
  }, [attempt]);

  return (
    <ServerStateContext.Provider value={{ ...value, refresh }}>
      {children}
    </ServerStateContext.Provider>
  );
}
