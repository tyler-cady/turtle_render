from setuptools import setup, find_packages

setup(
    name="turtle_render",
    version="1.0.0",
    description="AI image renderer using Python turtles",
    author="Tyler Cady",
    packages=find_packages(),
    install_requires=[
        "torch", 
        "diffusers", 
        "Pillow", 
        "opencv-python",  
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
