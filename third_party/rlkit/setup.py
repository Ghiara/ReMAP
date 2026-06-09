from pathlib import Path

from setuptools import setup


def discover_packages() -> list[str]:
    root = Path(__file__).resolve().parent
    packages = []

    for init_file in root.rglob("__init__.py"):
        rel_parent = init_file.parent.relative_to(root)
        if rel_parent == Path("."):
            packages.append("rlkit")
        else:
            packages.append("rlkit." + ".".join(rel_parent.parts))

    return sorted(set(packages))


setup(
    name="rlkit",
    version="0.0.0",
    description="Local editable rlkit package for Inference-reutilization-MRL",
    packages=discover_packages(),
    package_dir={"rlkit": "."},
    include_package_data=True,
    package_data={"rlkit": ["envs/assets/*.xml"]},
)
