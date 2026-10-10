/*
 Vergence - host-supplied policy

 What the depth pipeline needs but can't decide for itself, because the answer
 belongs to the app. Set once at launch; read from whatever thread the cache
 runs on (`DepthCacheStore.enforceBudget` is called from the conversion queue
 and from settings screens, neither of which can await), hence a lock.
 */

import Foundation

public enum VergencePolicy {
    /// Byte cap the depth cache must stay under, computed from its current
    /// total size. A host may return a cap *below* the current size to force
    /// a trim (low free space).
    ///
    /// Unset means unbounded: entries are never evicted for size. Silently
    /// evicting a user's expensive conversions because a host forgot to set a
    /// cap would be worse than growing.
    public static var depthCacheCap: (@Sendable (_ currentSize: Int64) -> Int64)? {
        get { storage.withLock { $0 } }
        set { storage.withLock { $0 = newValue } }
    }

    private static let storage = VergenceLock<(@Sendable (Int64) -> Int64)?>(nil)
}
