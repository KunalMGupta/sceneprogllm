from setuptools import setup, find_packages
setup(
    name='sceneprogllm',  # Replace with your package's name
    version='0.1.0',    # Replace with your package's version
    description='An LLM wrapper built for scene prog projects',
    long_description=open('README.md').read(),  # Optional: Use your README for a detailed description
    long_description_content_type='text/markdown',
    author='Kunal Gupta',
    author_email='k5upta@ucsd.edu',
    # url='https://github.com/yourusername/my_package',  # Optional: Replace with your repo URL
    packages=find_packages(),  # Automatically find all packages in your project
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',  # Replace with the minimum Python version your package supports
)