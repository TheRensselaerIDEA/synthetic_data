import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sythetic_data_kb22",
    version="0.0.1",
    author="Karan Bhanot",
    author_email="bhanotkaran22@gmail.com",
    description="",
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
    	"numpy",
    	"pandas",
    	"scipy",
    	"scikit-learn",
    	"tensorflow==1.13.1",
        "progress",
        "psutil",
        "tqdm",
        "matplotlib",
        "seaborn"
    ],
    python_requires='>=3.6',
)