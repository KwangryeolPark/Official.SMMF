import setuptools

AUTHOR=""
AUTHOR_EMAIL=""
URL=""

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name="smmf",
    version="1.0.0",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description="Implementation of SMMF",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    packages=setuptools.find_packages(),
    install_requires=[
        'torch>=1.7.0',
        'numpy'
    ]
)
