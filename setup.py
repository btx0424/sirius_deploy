from setuptools import setup, find_packages

setup(
    name="sirius_deploy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "h5py",
        "lcm",
        "mujoco",
        "onnxruntime",
    ],
    description="Deployment code for Sirius robot",
    author="Sirius Team",
    python_requires=">=3.8",
)
