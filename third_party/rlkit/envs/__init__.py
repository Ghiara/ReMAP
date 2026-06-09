import importlib
import os
import warnings


ENVS = {}


def register_env(name):
    """Registers a env by name for instantiation in rlkit."""

    def register_env_fn(fn):
        if name in ENVS:
            raise ValueError("Cannot register duplicate env {}".format(name))
        if not callable(fn):
            raise TypeError("env {} must be callable".format(name))
        ENVS[name] = fn
        return fn

    return register_env_fn


# Import env modules opportunistically so optional dependencies in unrelated
# environments do not prevent required envs from registering.
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        try:
            importlib.import_module(f'{__name__}.{module}')
        except Exception as exc:
            warnings.warn(
                f"Skipping optional env module {__name__}.{module}: {exc}",
                RuntimeWarning,
            )
