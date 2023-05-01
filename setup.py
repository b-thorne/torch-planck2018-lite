from setuptools import setup, find_packages

setup(
    name="torch_planck2018_lite",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    test_suite="tests",
)
