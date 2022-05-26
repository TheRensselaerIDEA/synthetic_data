import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="synthetic_data",
    version="1.0.0",
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
    	"numpy==1.19.5",
    	"pandas==1.1.5",
    	"scipy==1.5.4",
    	"scikit-learn==0.24.2",
    	"tensorflow==2.7.2",
        "psutil==5.8.0",
        "tqdm==4.17.0",
        "matplotlib==2.1.2",
        "seaborn==0.9.0"
    ],
    python_requires='>=3.6, <3.8',
)
