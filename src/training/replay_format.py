CURRENT_REPLAY_VERSION = "v2.1.0-perpetual-chase"


def is_supported_replay_version(version):
    return str(version).startswith("v2.1.")
