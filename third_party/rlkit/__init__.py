import sys

# Allow legacy imports like `import rlkit` to resolve to the vendored package.
sys.modules.setdefault('rlkit', sys.modules[__name__])
