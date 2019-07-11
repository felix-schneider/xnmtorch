from setuptools import setup

import xnmtorch

with open("requirements.txt") as req_file:
    requirements = [line.strip() for line in req_file]

setup(
    name="xnmtorch",
    version=xnmtorch.__version__,
    description="A neural toolkit",
    author="Felix Schneider",
    python_requires=">=3.6",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "xnmtorch = xnmtorch.xnmtorch_run_experiment:main",
            "xnmtorch-decode = xnmtorch.xnmtorch_decode:main"
        ]
    },
    extras_require={
        'bleu': ['sacrebleu'],
    }
)
