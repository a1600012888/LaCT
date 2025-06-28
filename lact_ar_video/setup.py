from setuptools import setup, find_packages

setup(
    name="minVid",
    version="0.1.0",
    description="Minimal Video Generation Library",
    packages=["minVid"],
    include_package_data=True,
    install_requires=[
        "torch>=2.5.0",
        "torchvision>=0.20.0",
        "numpy>=1.20.0",
        "Pillow>=9.0.0",
        "tqdm>=4.64.0",
        "imageio>=2.25.0",
        "imageio-ffmpeg>=0.4.7",
        "transformers>=4.27.0",
        "diffusers>=0.16.0",
        "huggingface-hub>=0.25.1",
    ],
    python_requires=">=3.8",
)