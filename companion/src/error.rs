use crate::message::FrameError;

/// Why a pairing ended without a grant, or a call didn't fit its state.
#[derive(Debug, thiserror::Error)]
pub enum PairingError {
    #[error("network: {0}")]
    Io(#[from] std::io::Error),
    #[error("malformed pairing message: {0}")]
    Frame(#[from] FrameError),
    /// The peer sent a message that doesn't belong at this point, or one that
    /// doesn't check out (a reveal that doesn't match its commitment, a bad key,
    /// a commit from another device than the one the listener was for).
    #[error("the other side broke the protocol: {0}")]
    Protocol(&'static str),
    #[error("the sealed grant couldn't be opened")]
    Seal(#[from] crate::exchange::SealError),
    #[error("pairing was declined on the host")]
    Denied,
    #[error("the pairing connection closed. Make sure both apps are up to date and try again")]
    Closed,
    #[error("the pairing timed out")]
    TimedOut,
    /// A call made in the wrong state (approving twice, approving before the
    /// code is known).
    #[error("not possible now: {0}")]
    State(&'static str),
}
