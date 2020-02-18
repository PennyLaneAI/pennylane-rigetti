#!/usr/bin/env python3
import sys
import os
from setuptools import setup

with open("pennylane_forest/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")


requirements = ["pyquil>=2.16", "pennylane>=0.8"]

info = {
    "name": "PennyLane-Forest",
    "version": version,
    "maintainer": "Rigetti Computing Inc.",
    "maintainer_email": "sohaib@rigetti.com",
    "url": "http://rigetti.com",
    "license": "BSD 3-Clause",
    "packages": ["pennylane_forest"],
    "entry_points": {
        "pennylane.plugins": [
            "forest.qpu = pennylane_forest:QPUDevice",
            "forest.qvm = pennylane_forest:QVMDevice",
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
    # 'extras_require': extra_requirements,
    "command_options": {
        "build_sphinx": {"version": ("setup.py", version), "release": ("setup.py", version)}
    },
}

classifiers = [
    "Development Status :: 3 - Alpha",
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
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Physics",
]

setup(classifiers=classifiers, **(info))
