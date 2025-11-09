"""
Setup script for the Fake Review Detection System
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fake-review-detection",
    version="1.0.0",
    author="Aravind S S",
    author_email="aravindss2004@example.com",
    description="AI-powered fake review detection using ensemble machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aravindss2004/fake-review-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "fake-review-detect=backend.app:main",
        ],
    },
)
