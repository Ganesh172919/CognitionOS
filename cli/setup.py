"""Setup script for CognitionOS CLI."""

from setuptools import setup, find_packages

setup(
    name="cognition-cli",
    version="1.0.0",
    description="CognitionOS management CLI tool",
    author="CognitionOS Team",
    packages=find_packages(),
    install_requires=[
        "httpx>=0.26.0",
    ],
    entry_points={
        "console_scripts": [
            "cognition-cli=cli.main:main",
        ],
    },
    python_requires=">=3.10",
)
