from setuptools import setup, find_packages

from kagu import __version__, __build__


with open("README.md", "r", encoding="utf-8") as f:
  long_description = f.read()

setup(
  name="kagu-torch",
  version=__version__,
  build=__build__,
  author="Ramy Mounir",
  url="https://ramymounir.com/docs/AutomatedEthogramming/",
  description=r"""Official implementation of Towards Automated Ethogramming: Cognitively-Inspired Event Segmentation for Streaming Wildlife Video Monitoring""",
  long_description=long_description,
  long_description_content_type="text/markdown",
  packages=find_packages(),
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: Free for non-commercial use",
  ],
  python_requires='>=3.10',
  install_requires=[
      'torch>=2.0.0',
      'ddpw>=5.1.1',
      'opencv-python-headless',
      'tqdm',
      'torchvision',
      'tensorboard',
      ]
)
