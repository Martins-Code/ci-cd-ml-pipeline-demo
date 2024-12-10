from setuptools import find_packages, setup
from typing import List

EDITABLE_INSTALL_ENTRY = '-e .'

def get_requirements(filepath: str) -> List[str]:
    '''
    This function will return the list of requirements
    from the provided requirements.txt file, excluding
    any development entries (like -e .).
    '''
    try:
        with open(filepath) as file_obj:
            requirements = [
                req.strip() for req in file_obj.readlines() if req.strip() != EDITABLE_INSTALL_ENTRY
            ]
            
    except FileNotFoundError:
        raise FileNotFoundError(f"{filepath} not found.")
    
    return requirements


setup(
    name='ci-cd-ml-pipeline-demo',
    version='0.0.1',
    author='Martins Okoye',
    author_email='Martinscode33@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)
