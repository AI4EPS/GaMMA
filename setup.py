from setuptools import setup

setup(
    name="GMMA",
    version="1.2.13",
    long_description="*GaMMA*: *Ga*ussian *M*ixture *M*odel *A*ssociation",
    long_description_content_type="text/markdown",
    packages=["gamma"],
    install_requires=["scikit-learn==1.6.1", "scipy", "numpy", "pyproj", "tqdm", "numba"],
)
