from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="PyNILC",  # Your package name
    version="0.1.0",  # Initial version; update as needed
    author="Anto Idicherian Lonappan",  # Replace with your name
    author_email="mail@antolonappan.me",  # Replace with your email
    description="A Python package for NILC methods in cosmology",
    long_description=long_description,
    long_description_content_type="text/markdown",  # If your README is in Markdown format
    url="https://github.com/echoCMB/PyNILC",  # URL of your GitHub repository
    packages=find_packages(),  # Automatically find packages in your project
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Adjust if your license differs
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",  # Specify the Python version you want to support
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "healpy",
        "ducc0",  # Add ducc0 as a dependency
    ],
    extras_require={
        "dev": ["pytest", "sphinx", "black"],  # Optional dependencies for development
    },
    entry_points={
        "console_scripts": [
            "pynilc=pynilc.cli:main",  # Example of creating a CLI tool; adjust if needed
        ]
    },
    include_package_data=True,  # Include non-Python files specified in MANIFEST.in
)