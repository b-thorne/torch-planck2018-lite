from setuptools import setup, find_packages

setup(
    name="torch-planck2018-lite",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
    ],
    include_package_data=True,
    package_data={"torch-planck2018-lite": ["data/*"]},
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    test_suite="tests",
)
