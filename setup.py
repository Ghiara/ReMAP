from pathlib import Path

from setuptools import setup


ROOT = Path(__file__).resolve().parent


def discover_packages(root: Path, top_level: str) -> list[str]:
    packages = []
    for init_file in root.rglob("__init__.py"):
        rel_parent = init_file.parent.relative_to(root)
        if rel_parent == Path("."):
            packages.append(top_level)
        else:
            packages.append(f"{top_level}." + ".".join(rel_parent.parts))
    return sorted(set(packages))


packages = []
package_dir = {}

package_roots = {
    "ReMAP": ROOT / "ReMAP",
    "smrl": ROOT / "third_party" / "Meta_RL" / "smrl",
    "specific": ROOT / "third_party" / "Meta_RL" / "specific",
    "meta_envs": ROOT / "third_party" / "Meta_RL" / "submodules" / "meta-environments-main" / "meta_envs",
    "mrl_analysis": ROOT / "third_party" / "Meta_RL" / "submodules" / "MRL-analysis-tools-main" / "mrl_analysis",
    "rlkit": ROOT / "third_party" / "Meta_RL" / "submodules" / "rlkit" / "rlkit",
    "bnpy": ROOT / "third_party" / "bnpy" / "bnpy",
}

for package_name, package_root in package_roots.items():
    packages.extend(discover_packages(package_root, package_name))
    package_dir[package_name] = str(package_root)


setup(
    name="remap",
    version="0.1.0",
    description="Editable root package for the ReMAP project",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=sorted(set(packages)),
    package_dir=package_dir,
    include_package_data=True,
    install_requires=[
        "click",
        "cloudpickle",
        "Cython>=0.29",
        "glfw",
        "gtimer",
        "gym==0.26.2",
        "gymnasium==0.29.1",
        "imageio",
        "imageio-ffmpeg",
        "ipython",
        "joblib==0.14.1",
        "jsonlines",
        "matplotlib",
        "memory-profiler",
        "metaworld",
        "mujoco-py==2.1.2.14",
        "munkres==1.0.12",
        "networkx",
        "psutil",
        "requests",
        "numpy",
        "opencv-python",
        "pandas",
        "Pillow",
        "pygame",
        "pyglet",
        "PyOpenGL",
        "pytz",
        "PyYAML",
        "six",
        "ray",
        "scikit-learn==1.0.2",
        "scipy",
        "seaborn",
        "stable-baselines3",
        "tensorboard",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pytest",
        ],
    },
)
