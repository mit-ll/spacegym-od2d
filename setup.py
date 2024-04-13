from setuptools import setup, find_packages

def get_version():
    path = "src/orbit_defender2d/version.py"
    with open(path) as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("VERSION"):
            return line.strip().split("=")[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")

setup(name="orbit_defender2d", 
      version=get_version(),
      packages=find_packages('src'),
      package_dir={'': 'src'},
      python_requires=">=3",
      install_requires=[
        "numpy==1.21.6",
        "gym==0.21.0", #requires older version of setuptools and pip listed in requirements.txt
        "torch==1.10.1",        
        "pettingzoo==1.15.0",
        "networkx",
        "matplotlib",
        "pyzmq",
        "tornado==6.1",
        "pygame==2.0.3",
        "bidict",
      ]
      )