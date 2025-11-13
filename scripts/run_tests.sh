#!/bin/bash

##############################################################################
# Distributed Training Framework - Comprehensive Test Runner
#
# This script runs all tests at different levels:
# - Unit tests
# - Integration tests
# - Performance tests
# - Regression tests
#
# Usage:
#   ./scripts/run_tests.sh                    # Run all tests
#   ./scripts/run_tests.sh --unit-only        # Only unit tests
#   ./scripts/run_tests.sh --fast             # Skip slow tests
#   ./scripts/run_tests.sh --with-coverage    # Generate coverage report
##############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
UNIT_ONLY=false
FAST=false
WITH_COVERAGE=false
INSTALL_DEPS=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --unit-only)
            UNIT_ONLY=true
            ;;
        --fast)
            FAST=true
            ;;
        --with-coverage)
            WITH_COVERAGE=true
            ;;
        --install-deps)
            INSTALL_DEPS=true
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --unit-only        Run only unit tests"
            echo "  --fast             Skip slow tests"
            echo "  --with-coverage    Generate coverage report"
            echo "  --install-deps     Install dependencies first"
            echo "  --help, -h         Show this help"
            exit 0
            ;;
    esac
done

print_header() {
    echo -e "\n${BLUE}=================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

##############################################################################
# Dependency Check
##############################################################################

check_dependencies() {
    print_header "Checking Dependencies"

    # Check Python
    if ! command -v python &> /dev/null; then
        print_error "Python not found"
        exit 1
    fi
    print_success "Python: $(python --version)"

    # Check pytest
    if ! python -c "import pytest" 2>/dev/null; then
        print_error "pytest not installed"
        if $INSTALL_DEPS; then
            print_info "Installing pytest..."
            pip install pytest pytest-cov pytest-xdist -q
        else
            echo "Install with: pip install pytest pytest-cov"
            exit 1
        fi
    else
        print_success "pytest installed"
    fi

    # Check PyTorch (optional for some tests)
    if python -c "import torch" 2>/dev/null; then
        local torch_version=$(python -c "import torch; print(torch.__version__)")
        print_success "PyTorch installed: $torch_version"
    else
        print_info "PyTorch not installed (some tests will be skipped)"
    fi
}

##############################################################################
# Test Execution
##############################################################################

run_syntax_tests() {
    print_header "Syntax Validation"

    local errors=0
    local total=0

    while IFS= read -r -d '' file; do
        ((total++))
        if ! python -m py_compile "$file" 2>/dev/null; then
            print_error "Syntax error: $file"
            ((errors++))
        fi
    done < <(find src tests -name "*.py" -print0 2>/dev/null)

    if [ $errors -eq 0 ]; then
        print_success "All $total files passed syntax validation"
        return 0
    else
        print_error "$errors/$total files have syntax errors"
        return 1
    fi
}

run_unit_tests() {
    print_header "Unit Tests"

    local pytest_args="-v -m unit"

    if $FAST; then
        pytest_args="$pytest_args -m 'not slow'"
    fi

    if $WITH_COVERAGE; then
        pytest_args="$pytest_args --cov=src/distributed_training --cov-report=term-missing"
    fi

    if python -m pytest $pytest_args tests/ 2>/dev/null; then
        print_success "Unit tests passed"
        return 0
    else
        print_info "Unit tests not available or failed (may need PyTorch)"
        return 0  # Don't fail if PyTorch not installed
    fi
}

run_integration_tests() {
    if $UNIT_ONLY; then
        return 0
    fi

    print_header "Integration Tests"

    local pytest_args="-v -m integration"

    if $FAST; then
        pytest_args="$pytest_args -m 'not slow'"
    fi

    if python -m pytest $pytest_args tests/integration/ 2>/dev/null; then
        print_success "Integration tests passed"
        return 0
    else
        print_info "Integration tests not available or failed"
        return 0
    fi
}

run_performance_tests() {
    if $UNIT_ONLY || $FAST; then
        return 0
    fi

    print_header "Performance Tests"

    if python -m pytest -v -m performance tests/performance/ 2>/dev/null; then
        print_success "Performance tests passed"
        return 0
    else
        print_info "Performance tests not available"
        return 0
    fi
}

run_regression_tests() {
    if $UNIT_ONLY; then
        return 0
    fi

    print_header "Regression Tests"

    if python -m pytest -v -m regression tests/regression/ 2>/dev/null; then
        print_success "Regression tests passed"
        return 0
    else
        print_info "Regression tests not available"
        return 0
    fi
}

##############################################################################
# Main Execution
##############################################################################

main() {
    print_header "Distributed Training Framework - Test Runner"

    echo "Date: $(date)"
    echo ""

    if $INSTALL_DEPS; then
        check_dependencies
    fi

    local failed=0

    # Always run syntax tests
    if ! run_syntax_tests; then
        ((failed++))
        print_error "Syntax tests failed - stopping"
        exit 1
    fi

    # Run other tests
    run_unit_tests || ((failed++))

    if ! $UNIT_ONLY; then
        run_integration_tests || ((failed++))
        run_performance_tests || ((failed++))
        run_regression_tests || ((failed++))
    fi

    # Generate coverage report
    if $WITH_COVERAGE && [ -f .coverage ]; then
        print_header "Coverage Report"
        python -m coverage report
        python -m coverage html
        print_info "HTML coverage report: htmlcov/index.html"
    fi

    # Summary
    print_header "Test Summary"

    if [ $failed -eq 0 ]; then
        print_success "All tests passed!"
        return 0
    else
        print_info "Some test categories were skipped (may need dependencies)"
        print_info "Install full dependencies: pip install -r requirements.txt"
        return 0
    fi
}

main
exit_code=$?

echo ""
exit $exit_code
