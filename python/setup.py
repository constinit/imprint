import os
import sys
from glob import glob
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, ParallelCompile
import subprocess

# Optional multithreaded build
ParallelCompile("NPY_NUM_BUILD_JOBS").install()

# Get bazel output base
process = subprocess.Popen(['bazel', 'info', 'output_base'],
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE)
stdout, _ = process.communicate()
BAZEL_OUTPUT_BASE = stdout.decode('utf-8').rstrip() # remove trailing whitespaces
BAZEL_EXTERNAL = os.path.join(BAZEL_OUTPUT_BASE, "external")

# Define some variables for ease of interpretation
CWD = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.join(CWD, '..')
EIGEN_INCLUDE_DIR = os.path.join(BAZEL_EXTERNAL, 'eigen')
print(f"eigen={EIGEN_INCLUDE_DIR}")
KTHOHR_STATS_INCLUDE_DIR = os.path.join(BAZEL_EXTERNAL, 'kthohr_stats/include')
KTHOHR_GCEM_INCLUDE_DIR = os.path.join(BAZEL_EXTERNAL, 'kthohr_gcem/include')
KEVLAR_INCLUDE_DIR = os.path.join(ROOT_DIR, 'kevlar/include')
PYKEVLAR_INCLUDE_DIR = os.path.join(CWD, 'src')

# Debug flag
# TODO: make this a command-line argument
DEBUG = False

# Get long description by reading README.md (as one should).
with open(os.path.join(CWD, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Get extra compiler/linker flags
if sys.platform == "win32":
    if DEBUG:
        extra_compile_args = ["/O0"]
        extra_link_args = []
    else:
        extra_compile_args = ["/openmp", "/O2"]
        extra_link_args = ["/openmp"]
else:
    if DEBUG:
        extra_compile_args = [
            "-g",
            "-O0",
            "--std=c++17",
        ]
        extra_link_args = []
    else:
        extra_compile_args = [
            "-g",
            "-O3",
            "-fopenmp",
            "-ffast-math",
            "--std=c++17",
        ]
        extra_link_args = ["-fopenmp"]

# Add extension module for pykevlar
ext_modules = [
    Pybind11Extension(
        "pykevlar.core",
        sorted(glob(PYKEVLAR_INCLUDE_DIR + '/**/*.cpp', recursive=True)),
        include_dirs=[
            EIGEN_INCLUDE_DIR,
            KTHOHR_STATS_INCLUDE_DIR,
            KTHOHR_GCEM_INCLUDE_DIR,
            KEVLAR_INCLUDE_DIR,
            PYKEVLAR_INCLUDE_DIR,
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
]

setup(
    name="pykevlar",
    # TODO: seems irrelevant
    #use_scm_version={"version_scheme": "post-release"},
    #setup_requires=["setuptools_scm"],
    description="Kevlar exports to Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mikesklar/kevlar",
    author="Confirm Solutions Modelling",
    author_email="contact@confirmsol.org",
    #TODO: lol we need one: license="BSD",
    classifiers=[  # Optional
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(),
    # TODO: doesn't seem relevant
    #entry_points=None
    #if os.environ.get("CONDA_BUILD")
    #else """
    #    [console_scripts]
    #    glm_benchmarks_run = glum_benchmarks.cli_run:cli_run
    #    glm_benchmarks_analyze = glum_benchmarks.cli_analyze:cli_analyze
    #""",
    ext_modules=ext_modules,
    zip_safe=False,
)
