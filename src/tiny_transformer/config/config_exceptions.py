# ---------------------------------------------------------------------------
# ConfigError
#
# ConfigError is used for problems with loading or validating configuration.
# This keeps config failures separate from other program errors.
# ---------------------------------------------------------------------------
class ConfigError(ValueError):
    pass
