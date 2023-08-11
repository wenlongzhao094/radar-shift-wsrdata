"""
Training scripts for multiple universe boxes.
"""
from setuptools import find_packages, setup

install_requires = [
    'wsrlib @ git+https://github.com/darkecology/pywsrlib#egg=wsrlib', # commit 3690123 tested
]

setup(
    name="wsrdata",
    version="0.1.0",
    description="Rendering datasets from weather surveillance radar scans. "
                "The datasets can be used to train and evaluate detection and tracking models.",
    packages=find_packages(include=["wsrdata"]),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.6.0",
    install_requires=install_requires,
    classifiers=[
        'Development Status :: 3 - Alpha',
    ]
)
