#!/usr/bin/env python3
import sys
import os
from setuptools import setup

with open("pennylane_forest/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")


requirements = ["pyquil>=3.0.0,<4.0.0", "qcs-api-client>=0.20.13<0.22.0", "pennylane>=0.18"]

info = {
    "name": "PennyLane-Forest",
    "version": version,
    "maintainer": "Xanadu Inc.",
    "maintainer_email": "software@xanadu.ai",
    "url": "https://github.com/PennyLaneAI/pennylane-forest",
    "license": "BSD 3-Clause",
    "packages": ["pennylane_forest"],
    "entry_points": {
        "pennylane.plugins": [
            "forest.qvm = pennylane_forest:QVMDevice",
            "forest.qpu = pennylane_forest:QPUDevice",
            "forest.wavefunction = pennylane_forest:WavefunctionDevice",
            "forest.numpy_wavefunction = pennylane_forest:NumpyWavefunctionDevice",
        ],
        "pennylane.io": [
            "pyquil_program = pennylane_forest:load_program",
            "quil = pennylane_forest:load_quil",
            "quil_file = pennylane_forest:load_quil_from_file",
        ],
    },
    "description": "Rigetti backend for the PennyLane library",
    "long_description": open("README.rst").read(),
    "provides": ["pennylane_forest"],
    "install_requires": requirements,
    "long_description_content_type": "text/x-rst"
}

classifiers = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD License",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Physics",
]

setup(classifiers=classifiers, **(info))
