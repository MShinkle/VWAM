from gettext import install
import os
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + '/requirements.txt'
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()
setuptools.setup(
    name="VWAM",
    install_requires=install_requires,
    version="0.0.1",
    author="Matthew Shinkle",
    author_email="mshinkle1040@gmail.com",
    description="Python/PyTorch code for Voxel-Wise Activation Maximization (VWAM)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MShinkle/VWAM",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    # python_requires='>=3.6',
)