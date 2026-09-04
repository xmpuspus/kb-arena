import { useEffect, useState } from "react";

/**
 * A counter that changes when the API token does.
 *
 * A page that already got 401 keeps showing the refusal until something reads
 * again. Entering a token is the moment to retry, or the reader sees no change
 * and concludes the token did not work. Put this in the dependency list of the
 * effect that does a protected read.
 */
export function useTokenEpoch(): number {
  const [epoch, setEpoch] = useState(0);

  useEffect(() => {
    const bump = () => setEpoch((n) => n + 1);
    window.addEventListener("kb-arena-token-changed", bump);
    return () => window.removeEventListener("kb-arena-token-changed", bump);
  }, []);

  return epoch;
}
