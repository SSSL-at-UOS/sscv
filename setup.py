import setuptools

setuptools.setup(
    name="sscv",
    version="0.1.12",
    license='MIT',
    author="Byunghyun Kim",
    author_email="kimbh.mail@gmail.com",
    description="Open computer vision libary for strucutral health monitoring",
    long_description=open('README.md').read(),
    url="https://github.com/SSSL-at-UOS/sscv",
    packages=setuptools.find_packages(),
    classifiers=[
        # 패키지에 대한 태그
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
)
