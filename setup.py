import os
from setuptools import setup, find_packages


cur_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(cur_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="DeePFAS",
    packages=find_packages(exclude=['__pycache__', 'lightning_logs']),
    version="1.0.0",
    description="DeePFAS (Deep Learning-Enabled Rapid Annotation of PFAS)",
    author="Wang Heng",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CMDM-Lab/DeePFAS",
    install_requires=[
        "beartype==0.18.5",
        "einops==0.6.1",
        "chempy==0.8.3",
        "matplotlib==3.8.2",
        "molmass==2024.5.24",
        "numpy==1.26.4",
        "pandas==2.2.1",
        "pydantic==2.10.6",
        "pyteomics==4.6.3",
        "rdkit==2023.9.4",
        "scikit-learn>=1.5.0",
        "scipy==1.12.0",
        "torch>=2.7.1-rc1",
        "tqdm==4.66.1",
        "seaborn==0.13.2",
        "h5py==3.12.1",
        "wandb==0.19.5",
        "pytorch-lightning==2.5.0.post0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)