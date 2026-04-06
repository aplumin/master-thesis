from setuptools import find_packages, setup

setup(
    name="master-thesis",
    author="Aaron Plumin",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "diffrax==0.7.2",
        "jax=0.7.2",
        "jupyter=1.1.1",
        "numba==0.64.0",
        "seaborn=0.13.2",
        "snakemake==9.19.0",
    ],
    python_requires=">=3.11",
)
