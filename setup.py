from setuptools import setup, find_packages  # pyright: ignore[reportMissingModuleSource]

HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str):
    requirements = []
    with open(file_path, "r") as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name="customer_churn_prediction",
    version="0.0.1",
    author="Gourav Chhatwani",
    author_email="Chhatwanigourav@gmail.com",
    description="End-to-End Customer Churn Prediction using Machine Learning",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    include_package_data=True,
    python_requires=">=3.11",
)