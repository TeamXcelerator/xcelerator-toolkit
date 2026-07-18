use std::error::Error;
use std::fmt::{Display, Formatter};

/// Invalid public configuration.  Configuration errors are kept separate from
/// numerical non-convergence so callers can fail before expensive work begins.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConfigError {
    message: String,
}

impl ConfigError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl Display for ConfigError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for ConfigError {}
