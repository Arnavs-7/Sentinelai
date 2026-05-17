from pathlib import Path

from setuptools import find_packages, setup

here = Path(__file__).parent
long_description = ""
readme = here / "README.md"
if readme.exists():
    long_description = readme.read_text(encoding="utf-8")

requirements = []
req_file = here / "requirements.txt"
if req_file.exists():
    requirements = [
        line.strip()
        for line in req_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="sentinel-ai",
    version="1.0.0",
    description="Sentinel AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
    include_package_data=True,
)
