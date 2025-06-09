#!/usr/bin/env python3
"""Setup script for Tucan: A Function-Calling Evaluation Framework"""

from setuptools import setup, find_packages

# Read the README file for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# Read requirements from requirements.txt (excluding commented unsloth)
def get_requirements():
    requirements = []
    with open("requirements.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("# unsloth"):
                requirements.append(line)
    return requirements


setup(
    name="tucan-eval",
    version="1.0.0",
    author="Your Name",  # Update this
    author_email="your.email@example.com",  # Update this
    description="A Function-Calling Evaluation Framework for Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/tucan",  # Update this
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tucan=tucan.__main__:main",
        ],
    },
    include_package_data=True,
    package_data={
        "tucan": ["*.yaml", "*.yml"],
    },
)
