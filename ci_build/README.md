# CI Build Folder

## Table of Contents
1. [PR-Checks Workflow File](#workflow-file) (.github/workflows/pr-checks.yml)
2. [pylintrc Configuration](#pylintrc-configuration) (ci_build/pylintrc))

---

## PR-Checks Workflow File

As part of our commitment to maintaining a high-quality codebase, we've set up a comprehensive CI pipeline through our PR-Checks workflow. This file automates important tasks, ensuring that every pull request meets the standards and is free from potential issues before merging.

### Overview

The workflow is triggered on all pull requests to the `main` branch. It consists of several jobs, each designed to validate different aspects of the code. The sequence begins by identifying the files that have changed, then checking for the presence of any PDF or Word documents, and finally running tests on Python files and Jupyter Notebooks. Here’s a detailed look at what each job does:

### 1. **File Checks**
   - **Purpose:** To verify that no unwanted files (such as Word or PDF documents) are included in the pull request.
   - **Process:** This job uses command-line utilities to search for `.docx` and `.pdf` files in the PR. If any are found, the job fails, prompting the contributor to remove these files before resubmitting the PR.

### 2. **Linting**
   - **Purpose:** To ensure that the code remains clean, readable, and adheres to PEP 8 standards.
   - **Tools:** 
     - [Pylint](https://www.pylint.org/)
     - [Flake8](https://flake8.pycqa.org/en/latest/)
   - **Process:** The linting job focuses on checking all `.py` and `.ipynb` files. It validates naming conventions, enforces the use of docstrings for functions, and detects errors. This step is crucial for maintaining consistent coding practices across the team.

### 3. **Security Tests**
   - **Purpose:** To ensure that the code is free from common security vulnerabilities.
   - **Tool:** [Bandit](https://bandit.readthedocs.io/en/latest/)
   - **Process:** We perform a security check using Bandit, which scans the codebase for potential security issues. This helps in proactively addressing any vulnerabilities before they become a problem.

---

## pylintrc Configuration

The `pylintrc` file is used to customise the linting process, allowing to enforce specific rules and ignore certain warnings that may not be relevant to the current project setup.

### Key Configurations

1. **Message Control**
   - **Purpose:** To fine-tune which types of messages (errors, warnings, etc.) are enabled or disabled during the linting process.
   - **Example:**
     ```ini
     [MESSAGES CONTROL]
     disable=E0401,C0114,F
     ```
   - **Explanation:** We’ve chosen to disable certain messages to better align with the current development phase:
     - **E0401/import-error:** Disabled because we haven't set up the environment yet, and we’re not checking module imports.
     - **C0114/missing-module-docstring:** Disabled since our projects have varying requirements, making module docstrings less relevant.
     - **F/Fatal Error Messages:** Disabled because we’re not currently validating if the modules can be built.

### Customisation

The `pylintrc` file is fully customisable to fit the unique needs of the project. As the project progresses, this file will must adjusted to reflect any changes in coding standards or project requirements. For a more in-depth understanding of the available options, refer to the [Pylint documentation](https://pylint.readthedocs.io/en/latest/index.html).

---

This markdown file serves as a guide to the current CI setup and linting configuration. As the project evolves, so will the processes, and this document must be updated accordingly. If you have any suggestions for improvements or additional checks, feel free to contribute!