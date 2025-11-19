from setuptools import setup, find_packages

setup(
    name="manga_ocr_dev",
    version="0.1.0",
    packages=['manga_ocr_dev', 'manga_ocr_dev.synthetic_data_generator_ko'],
    package_dir={'manga_ocr_dev': '.'},
    py_modules=['manga_ocr_dev.env'],
)