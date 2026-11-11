from setuptools import setup, find_packages

setup(
    name="regiomanip",
    version="1.0.0",
    description=(
        "Region-Level Affordances to Wrench-Bounded Action Constraints "
        "for Zero-Shot Tool Manipulation"
    ),
    author="RegioManip Authors",
    python_requires=">=3.10",
    packages=find_packages(exclude=["GroundingDINO*", "JAKA*", "experiment*", "analysis*"]),
    include_package_data=True,
    package_data={
        "regiomanip": ["configs/*.json"],
    },
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "opencv-python>=4.8",
        "Pillow>=10.0",
        "scikit-image>=0.21",
        "open3d>=0.16.0",
        "timm>=0.9",
        "openai>=1.10",
        "httpx>=0.25",
        "pyyaml>=6.0",
        "omegaconf>=2.3",
        "einops>=0.7",
    ],
    entry_points={
        "console_scripts": [
            "regiomanip=regiomanip.RegioManip2:main",
        ],
    },
)
