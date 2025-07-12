from setuptools import setup, find_packages

setup(
    name='aurora_trinity',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Flask',
        # Añade otras dependencias si las tienes
    ],
    entry_points={
        'console_scripts': [
            'aurora-app=frontend.app:main', # Si quisieras correrlo desde consola
        ],
    },
)
