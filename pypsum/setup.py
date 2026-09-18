from setuptools import setup

setup(
    name="pypsum",
    version="0.9",
    packages=["pypsum"],
    package_dir={"pypsum": "."},
    package_data={"pypsum": ["bin/*.so", "bin/*.pyd"]},
)