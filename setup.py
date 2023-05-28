import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mcts",
    version="0.1.0-alpha1",
    author="Peter Sanders",
    description="Monte-Carlo Tree Search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hxtk/mcts",
    packages=setuptools.find_packages(),
    install_requires=[
        "tensorflow>=2.11.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
