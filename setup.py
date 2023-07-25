from setuptools import setup

setup(
    name="GMMA",
    version="1.2.2",
    long_description="*GaMMA*: *Ga*ussian *M*ixture *M*odel *A*ssociation",
    long_description_content_type="text/markdown",
    packages=["gamma"],
    install_requires=["scikit-learn", "scipy", "numpy", "pyproj", "tqdm", "numba"],
)
