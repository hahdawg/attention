from setuptools import setup, find_packages

setup(
    name="attention",
    description="",
    author="Andrew Hah",
    author_email="hahdawg@yahoo.com",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "tokenizers",
        "torch"
    ],
    zip_safe=False
)
