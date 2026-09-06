import { useEffect } from "react";

/**
 * Drop the data one scope owns the moment the scope changes.
 *
 * A corpus picker changes the name over a table, a map or a rating, and the
 * read that fills them lands later. Whatever the last scope left on screen
 * reads as the new scope's data for that whole gap. Ordering the responses
 * does not close it, because nothing clears when the read starts.
 *
 * Call this above the effect that reads, so the clear runs first. The `clear`
 * callback also raises the page's own pending flag, because "empty" and "not
 * read yet" are two different answers.
 */
export function useScopeReset(scope: string, clear: () => void): void {
  useEffect(() => {
    clear();
    // `clear` is a new closure on every render, so listing it here would run
    // the reset on every render instead of on a scope change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scope]);
}
