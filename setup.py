import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="synthetic_data",
    version="0.9.1",
    author="Karan Bhanot",
    author_email="bhanotkaran22@gmail.com",
    description="Package that enables generation of synthetic data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TheRensselaerIDEA/synthetic_data",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    install_requires=[
    	"numpy==1.17.0",
    	"pandas",
    	"scipy",
    	"scikit-learn",
    	"tensorflow==2.3.1",
        "psutil",
        "tqdm",
        "matplotlib",
        "seaborn"
    ],
    python_requires='>=3.6, <3.8',
)
