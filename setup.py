#!/usr/bin/env python3
import os
import sys

from setuptools import setup

with open("pennylane_rigetti/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

with open("./requirements.txt") as f:
    requirements = [req.strip() for req in f.readlines()]

info = {
    "name": "PennyLane-Rigetti",
    "version": version,
    "maintainer": "Xanadu Inc.",
    "maintainer_email": "software@xanadu.ai",
    "url": "https://github.com/PennyLaneAI/pennylane-rigetti",
    "license": "BSD 3-Clause",
    "packages": ["pennylane_rigetti"],
    "entry_points": {
        "pennylane.plugins": [
            "rigetti.qvm = pennylane_rigetti:QVMDevice",
            "rigetti.qpu = pennylane_rigetti:QPUDevice",
            "rigetti.wavefunction = pennylane_rigetti:WavefunctionDevice",
            "rigetti.numpy_wavefunction = pennylane_rigetti:NumpyWavefunctionDevice",
        ],
        "pennylane.io": [
            "pyquil_program = pennylane_rigetti:load_program",
            "quil = pennylane_rigetti:load_quil",
            "quil_file = pennylane_rigetti:load_quil_from_file",
        ],
    },
    "description": "Rigetti backend for the PennyLane library",
    "long_description": open("README.rst").read(),
    "provides": ["pennylane_rigetti"],
    "install_requires": requirements,
    "long_description_content_type": "text/x-rst",
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
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Physics",
]

setup(classifiers=classifiers, **(info))
