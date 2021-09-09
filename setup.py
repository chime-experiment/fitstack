from setuptools import setup, find_packages

import versioneer

setup(
    name="fitstack",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        "draco",
        "emcee",
    ],
    author="The CHIME Collaboration",
    description="Fit eBOSSxCHIME stacked data.",
    license="MIT",
    url="https://github.com/chime-experiment/fitstack",
)
