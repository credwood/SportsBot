from setuptools import setup, find_packages

setup(
    name='SportsBot',
    version='1.0.0',
    url='https://github.com/credwood/SportsBot.git',
    author='Charysse Redwood',
    author_email='charysse.redwood@gmail.com',
    description='Sentiment analysis with Twitter conversations; collectoion, fine-tuning and predicting',
    packages=find_packages(),
    install_requires=[
        'numpy==1.18.5',
        'scipy==1.4.1',
        'torch==1.7.0',
        'torchvision=0.8.1',
        'transformers==3.4.0',
        'tweepy==3.9.0',
        'jsonlines==1.2.0',
        'dataclasses_json==0.5.2',
        'datasets==1.1.2',
        'matplotlib==3.3.3',
        'seaborn==0.11.1'
    ],
)
