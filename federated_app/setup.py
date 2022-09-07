from setuptools import setup, find_packages


setup(
    name="federated_learning_framework",
    version="1.0.0",
    author="Rares Patrascu",
    description="",
    url="",
    packages=find_packages(),
    python_requires=">3.7.5",
    install_requires=[
            'flwr==0.19.0',
            'tensorflow==2.8.0',
            'Flask==2.0.2',
            'flake8==4.0.1',
            'torch==1.11.0',
            'opencv-python == 4.4.0.46',
            'nptyping == ^1.4.4',
            'jsonargparse==4.7.0',
            'matplotlib==3.5.2'],
    entry_points={
        'console_scripts': [
            'run_federated_server= federated_app.federated_server.server.server:run'
            'convert_dataset=federated_app.federated_dataset.convertor.dataset_convertor:run']
    }
)
