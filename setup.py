from setuptools import setup, find_packages

setup(
    name='SportsBot',
    version='1.0.0',
    url='https://github.com/credwood/SportsBot.git',
    author='Charysse Redwood',
    author_email='charysse.redwood@gmail.com',
    description='Twitter conversation collector and classifier',
    packages=find_packages(),
    install_requires=[
        'numpy==1.18.5',
        'scipy==1.4.1',
        'tensorflow==2.3.0',
        'transformers==3.2.0',
        'tweepy==3.9.0'
    ],
)
