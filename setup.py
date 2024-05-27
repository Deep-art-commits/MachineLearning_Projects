from setuptools import find_packages,setup
from typing import List 
HYPHEN_DOT="-e ."
def get_requirements(filepath:str)->List[str]:
    
    '''
    This function returns the list of requirements 
    '''
    with open(filepath) as fileobj:
        requirements=fileobj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if HYPHEN_DOT in requirements:
            requirements.remove(HYPHEN_DOT)
    return requirements
setup(
name="MachimeLearning_Projects",
version="0.0.1",
author="Dilpreet Singh",
author_email="dilpreet.singh12@outlook.com",
packages=find_packages(),
install_requires=get_requirements("Requirement.txt")
)