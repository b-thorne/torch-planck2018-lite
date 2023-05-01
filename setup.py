from setuptools import setup, find_packages

setup(
    name="torch-planck2018-lite",
    version="0.1",
    author="Benjamin Thorne",
    author_email="bn.thorne@gmail.com",
    description="A PyTorch implementation of the Planck 2018 Lite likelihood for the Cosmic Microwave Background (CMB) power spectra.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/torch-planck2018-lite",
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
