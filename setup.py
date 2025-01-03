from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name='UniRetrieval',
    version='0.0.1',
    description='Toolkit for Universal Retrieval, such as text retreival, item recommendation, image retrieval, etc.',
    long_description=readme,
    long_description_content_type="text/markdown",
    author_email='liandefu@ustc.edu.cn',
    url='https://github.com/USTCLLM/UniRetrieval',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch>=1.6.0',
        'transformers==4.44.2',
        'datasets==2.19.0',
        'accelerate>=0.20.1',
        'sentence_transformers',
        'peft',
        'ir-datasets',
        'sentencepiece',
        'protobuf'
    ],
)
