"""
Minimal Local Tests - No Dependencies Required

This test module can run without PyTorch or other heavy dependencies.
It validates:
- Project structure
- Configuration files
- Basic Python imports
- Documentation completeness

Usage:
    python tests/test_minimal_local.py

Or with pytest:
    pytest tests/test_minimal_local.py -v
"""

import os
import sys
import importlib.util
from pathlib import Path


class TestProjectStructure:
    """Test project structure and file organization"""

    def test_required_directories_exist(self):
        """Verify all required directories exist"""
        required_dirs = [
            "src/distributed_training",
            "src/distributed_training/data_parallel",
            "src/distributed_training/model_parallel",
            "src/distributed_training/pipeline_parallel",
            "src/distributed_training/advanced",
            "src/distributed_training/checkpointing",
            "src/distributed_training/data_loading",
            "src/distributed_training/fault_tolerance",
            "src/distributed_training/profiling",
            "src/distributed_training/optimization",
            "src/distributed_training/visualization",
            "src/distributed_training/config",
            "src/distributed_training/cli",
            "src/distributed_training/benchmarking",
            "src/distributed_training/utils",
            "tests",
            "tests/performance",
            "tests/regression",
            "tests/integration",
            "tests/advanced",
            "examples",
            "docs",
            "scripts",
        ]

        missing = []
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                missing.append(dir_path)

        assert len(missing) == 0, f"Missing directories: {missing}"
        print(f"✓ All {len(required_dirs)} required directories exist")

    def test_required_files_exist(self):
        """Verify all required files exist"""
        required_files = [
            "README.md",
            "requirements.txt",
            "setup.py",
            "pytest.ini",
            "LICENSE",
            "QUICKSTART.md",
            "MASTER_GUIDE.md",
            "PROJECT_SUMMARY.md",
            "ENHANCEMENTS_IMPLEMENTED.md",
            "src/distributed_training/__init__.py",
            "tests/TEST_README.md",
        ]

        missing = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing.append(file_path)

        assert len(missing) == 0, f"Missing files: {missing}"
        print(f"✓ All {len(required_files)} required files exist")

    def test_init_files_exist(self):
        """Verify __init__.py files exist in all packages"""
        package_dirs = [
            "src/distributed_training",
            "src/distributed_training/data_parallel",
            "src/distributed_training/model_parallel",
            "src/distributed_training/pipeline_parallel",
            "src/distributed_training/advanced",
            "src/distributed_training/checkpointing",
            "src/distributed_training/data_loading",
            "src/distributed_training/fault_tolerance",
            "src/distributed_training/profiling",
            "src/distributed_training/optimization",
            "src/distributed_training/visualization",
            "src/distributed_training/config",
            "src/distributed_training/cli",
            "src/distributed_training/benchmarking",
            "src/distributed_training/utils",
        ]

        missing = []
        for pkg_dir in package_dirs:
            init_file = os.path.join(pkg_dir, "__init__.py")
            if not os.path.exists(init_file):
                missing.append(init_file)

        assert len(missing) == 0, f"Missing __init__.py files: {missing}"
        print(f"✓ All {len(package_dirs)} packages have __init__.py")


class TestDocumentation:
    """Test documentation completeness"""

    def test_readme_not_empty(self):
        """Verify README has content"""
        with open("README.md", "r") as f:
            content = f.read()

        assert len(content) > 100, "README is too short"
        assert "Distributed Training" in content, "README missing main title"
        print(f"✓ README.md has {len(content)} characters")

    def test_all_docs_have_content(self):
        """Verify all documentation files have content"""
        doc_files = [
            "README.md",
            "QUICKSTART.md",
            "MASTER_GUIDE.md",
            "PROJECT_SUMMARY.md",
            "ENHANCEMENTS_IMPLEMENTED.md",
            "tests/TEST_README.md",
        ]

        for doc_file in doc_files:
            with open(doc_file, "r") as f:
                content = f.read()

            assert len(content) > 50, f"{doc_file} is too short or empty"
            print(f"✓ {doc_file}: {len(content)} characters")

    def test_documentation_total_lines(self):
        """Count total documentation lines"""
        doc_files = [
            "README.md",
            "QUICKSTART.md",
            "MASTER_GUIDE.md",
            "PROJECT_SUMMARY.md",
            "ENHANCEMENTS_IMPLEMENTED.md",
            "ENHANCEMENT_PLAN.md",
            "tests/TEST_README.md",
            "tests/TEST_SUMMARY.md",
        ]

        total_lines = 0
        for doc_file in doc_files:
            if os.path.exists(doc_file):
                with open(doc_file, "r") as f:
                    lines = len(f.readlines())
                    total_lines += lines

        print(f"✓ Total documentation: {total_lines} lines")
        assert total_lines > 1000, f"Expected >1000 lines of documentation, got {total_lines}"


class TestCodeStructure:
    """Test code organization and structure"""

    def test_python_files_syntax(self):
        """Verify all Python files have valid syntax"""
        import py_compile

        errors = []
        total = 0

        for root, dirs, files in os.walk("src"):
            for file in files:
                if file.endswith(".py"):
                    total += 1
                    filepath = os.path.join(root, file)
                    try:
                        py_compile.compile(filepath, doraise=True)
                    except py_compile.PyCompileError as e:
                        errors.append((filepath, str(e)))

        assert len(errors) == 0, f"Syntax errors in: {[e[0] for e in errors]}"
        print(f"✓ All {total} source files have valid syntax")

    def test_test_files_syntax(self):
        """Verify all test files have valid syntax"""
        import py_compile

        errors = []
        total = 0

        for root, dirs, files in os.walk("tests"):
            for file in files:
                if file.endswith(".py"):
                    total += 1
                    filepath = os.path.join(root, file)
                    try:
                        py_compile.compile(filepath, doraise=True)
                    except py_compile.PyCompileError as e:
                        errors.append((filepath, str(e)))

        assert len(errors) == 0, f"Syntax errors in: {[e[0] for e in errors]}"
        print(f"✓ All {total} test files have valid syntax")

    def test_example_files_syntax(self):
        """Verify all example files have valid syntax"""
        import py_compile

        if not os.path.exists("examples"):
            print("⚠ No examples directory found")
            return

        errors = []
        total = 0

        for root, dirs, files in os.walk("examples"):
            for file in files:
                if file.endswith(".py"):
                    total += 1
                    filepath = os.path.join(root, file)
                    try:
                        py_compile.compile(filepath, doraise=True)
                    except py_compile.PyCompileError as e:
                        errors.append((filepath, str(e)))

        assert len(errors) == 0, f"Syntax errors in: {[e[0] for e in errors]}"
        if total > 0:
            print(f"✓ All {total} example files have valid syntax")
        else:
            print("ℹ No example files found")


class TestConfiguration:
    """Test configuration files"""

    def test_requirements_txt_exists(self):
        """Verify requirements.txt exists and has content"""
        assert os.path.exists("requirements.txt"), "requirements.txt not found"

        with open("requirements.txt", "r") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        assert len(lines) > 0, "requirements.txt is empty"
        print(f"✓ requirements.txt has {len(lines)} dependencies")

    def test_setup_py_exists(self):
        """Verify setup.py exists and has basic structure"""
        assert os.path.exists("setup.py"), "setup.py not found"

        with open("setup.py", "r") as f:
            content = f.read()

        assert "setup(" in content, "setup.py missing setup() call"
        assert "name=" in content, "setup.py missing package name"
        print("✓ setup.py has valid structure")

    def test_pytest_ini_exists(self):
        """Verify pytest.ini exists and has markers"""
        assert os.path.exists("pytest.ini"), "pytest.ini not found"

        with open("pytest.ini", "r") as f:
            content = f.read()

        assert "markers" in content, "pytest.ini missing markers section"
        assert "testpaths" in content, "pytest.ini missing testpaths"
        print("✓ pytest.ini configured correctly")


class TestCodeMetrics:
    """Test code metrics and statistics"""

    def test_count_source_files(self):
        """Count source files"""
        count = 0
        for root, dirs, files in os.walk("src"):
            for file in files:
                if file.endswith(".py"):
                    count += 1

        print(f"✓ Source files: {count}")
        assert count > 20, f"Expected >20 source files, got {count}"

    def test_count_test_files(self):
        """Count test files"""
        count = 0
        for root, dirs, files in os.walk("tests"):
            for file in files:
                if file.endswith(".py") and file.startswith("test_"):
                    count += 1

        print(f"✓ Test files: {count}")
        assert count > 5, f"Expected >5 test files, got {count}"

    def test_count_lines_of_code(self):
        """Count total lines of code"""
        total_lines = 0

        for root, dirs, files in os.walk("src"):
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    with open(filepath, "r") as f:
                        total_lines += len(f.readlines())

        print(f"✓ Source code: {total_lines} lines")
        assert total_lines > 1000, f"Expected >1000 lines of source code, got {total_lines}"


def run_minimal_tests():
    """Run all minimal tests"""
    print("\n" + "=" * 70)
    print("Distributed Training Framework - Minimal Local Tests")
    print("=" * 70 + "\n")

    test_classes = [
        TestProjectStructure,
        TestDocumentation,
        TestCodeStructure,
        TestConfiguration,
        TestCodeMetrics,
    ]

    total_passed = 0
    total_failed = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 70)

        test_instance = test_class()
        test_methods = [m for m in dir(test_instance) if m.startswith("test_")]

        for method_name in test_methods:
            try:
                method = getattr(test_instance, method_name)
                method()
                total_passed += 1
            except AssertionError as e:
                print(f"✗ {method_name}: {e}")
                total_failed += 1
            except Exception as e:
                print(f"✗ {method_name}: Unexpected error: {e}")
                total_failed += 1

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Total: {total_passed + total_failed}")

    if total_failed == 0:
        print("\n✓ All minimal tests passed!")
        return 0
    else:
        print(f"\n✗ {total_failed} test(s) failed")
        return 1


if __name__ == "__main__":
    # Can be run directly without pytest
    exit_code = run_minimal_tests()
    sys.exit(exit_code)
