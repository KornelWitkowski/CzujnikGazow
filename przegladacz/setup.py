import setuptools


setuptools.setup(
    name="przegladacz",
    packages=setuptools.find_packages(),
    install_requires=[
        "tensorflow==2.5.0",
        "numpy==1.19.5",
        "pandas==1.3.1",
        "matplotlib==3.4.2",
        "PyQt5",
    ],
    extras_require={
        "test": [
            "black==21.6b0",
            "flake8==3.9.2",
            "pytest==6.2.4",
        ]
    },
    python_requires=">=3.8.0",
)