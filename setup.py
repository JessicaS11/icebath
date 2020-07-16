import setuptools

with open("README.rst", "r") as f:
    LONG_DESCRIPTION = f.read()

with open("requirements.txt") as f:
    INSTALL_REQUIRES = f.read().strip().split("\n")

setuptools.setup(
    name="icebath",
    version="0.0.1",
    author="Jessica Scheick",
    author_email="jbscheick@gmail.com",
    maintainer="Jessica Scheick",
    maintainer_email="jbscheick@gmail.com",
    description="inferring bathymetry from icebergs",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/x-rst",
    url="https://github.com/JessicaS11/icebath.git",
    license="BSD 3-Clause",
    packages=setuptools.find_packages(exclude=["*tests"]),
    install_requires=INSTALL_REQUIRES,
    python_requires=">=3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Development Status :: 1 - Planning",
    ],
)
