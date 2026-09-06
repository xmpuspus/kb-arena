"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { fetchServerStatus } from "@/lib/api";

/**
 * The one state every route names.
 *
 * `checking` until /health answers. `unreachable` when it never does, which is
 * not the same as a live server. The live states come from what /health
 * reports: an operator who published a read-only demo sets demo mode, the app
 * turns the same flag on for a machine with no model key, and the server says
 * whether this browser reached it over the loopback address. Reading demo mode
 * alone called a machine a hosted demo, and called every other one the
 * reader's own.
 */
export type ServerState =
  | "checking"
  | "unreachable"
  | "hosted-read-only"
  | "live-local"
  | "live-remote"
  | "live-unknown";

export interface ServerStateValue {
  state: ServerState;
  // True when /chat, arena matches, graph builds and the tools answer 503.
  writesOff: boolean;
  // True when the server says it turned demo mode on for itself, which means
  // no model key. An answer that does not say leaves this false, and the
  // banner then makes no claim about why the writes are off.
  keyless: boolean;
  refresh: () => void;
}

const FALLBACK: ServerStateValue = {
  state: "checking",
  writesOff: false,
  keyless: false,
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
    keyless: false,
  });
  const [attempt, setAttempt] = useState(0);

  const refresh = useCallback(() => {
    setValue({ state: "checking", writesOff: false, keyless: false });
    setAttempt((n) => n + 1);
  }, []);

  useEffect(() => {
    let active = true;
    fetchServerStatus().then((status) => {
      if (!active) return;
      if (!status) {
        setValue({ state: "unreachable", writesOff: false, keyless: false });
        return;
      }
      setValue({
        // Demo mode the app turned on for itself says nothing about hosting.
        // It says this machine has no model key, so live runs are off. Where
        // the server runs is the server's own answer, not a guess from a flag,
        // and an answer that skipped the flag says nothing either way.
        state:
          status.demoMode && status.demoModeAuto === false
            ? "hosted-read-only"
            : status.callerIsLocal === true
              ? "live-local"
              : status.callerIsLocal === false
                ? "live-remote"
                : "live-unknown",
        writesOff: status.demoMode,
        keyless: status.demoModeAuto === true,
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
