#!/usr/bin/env python3
"""
Comprehensive Test Runner for OOP Refactored Code

This script runs all tests for the OOP refactored codebase and provides
detailed reporting on test coverage and performance.
"""

import pytest
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
import subprocess

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestRunner:
    """Comprehensive test runner for OOP refactored code"""
    
    def __init__(self):
        self.project_root = project_root
        self.test_results = {}
        self.start_time = None
        self.end_time = None
    
    def run_all_tests(self):
        """Run all tests and generate comprehensive report"""
        print("=" * 80)
        print("COMPREHENSIVE OOP REFACTORED CODE TEST SUITE")
        print("=" * 80)
        print(f"Project Root: {self.project_root}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        self.start_time = time.time()
        
        # Test categories to run
        test_categories = [
            {
                'name': 'Core Component Tests',
                'path': 'tests/core/',
                'description': 'Test core OOP components and base classes'
            },
            {
                'name': 'Factory Pattern Tests',
                'path': 'tests/factories/',
                'description': 'Test factory pattern implementations'
            },
            {
                'name': 'Strategy Pattern Tests',
                'path': 'tests/strategies/',
                'description': 'Test strategy pattern implementations'
            },
            {
                'name': 'Implementation Tests',
                'path': 'tests/implementations/',
                'description': 'Test concrete implementations'
            },
            {
                'name': 'Integration Tests',
                'path': 'tests/integration/',
                'description': 'Test integration and end-to-end workflows'
            },
            {
                'name': 'Unit Tests',
                'path': 'tests/unit/',
                'description': 'Test individual components in isolation'
            }
        ]
        
        # Run each test category
        for category in test_categories:
            self.run_test_category(category)
        
        # Run existing tests for comparison
        self.run_existing_tests()
        
        self.end_time = time.time()
        
        # Generate comprehensive report
        self.generate_report()
    
    def run_test_category(self, category):
        """Run tests for a specific category"""
        print(f"\n{'='*60}")
        print(f"RUNNING: {category['name']}")
        print(f"Description: {category['description']}")
        print(f"Path: {category['path']}")
        print('='*60)
        
        category_start_time = time.time()
        
        try:
            # Run pytest for the category
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 
                category['path'],
                '-v', '--tb=short',
                '--json-report', '--json-report-file=pytest_report.json'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            category_end_time = time.time()
            duration = category_end_time - category_start_time
            
            # Parse results
            self.test_results[category['name']] = {
                'exit_code': result.returncode,
                'duration': duration,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'path': category['path'],
                'description': category['description']
            }
            
            # Print summary
            if result.returncode == 0:
                print(f"✅ {category['name']} - PASSED ({duration:.2f}s)")
            else:
                print(f"❌ {category['name']} - FAILED ({duration:.2f}s)")
                print(f"   Exit Code: {result.returncode}")
                
                # Print error summary
                if result.stderr:
                    print("   Errors:")
                    for line in result.stderr.split('\n')[-10:]:  # Last 10 lines
                        if line.strip():
                            print(f"   {line}")
            
        except Exception as e:
            print(f"❌ {category['name']} - ERROR: {e}")
            self.test_results[category['name']] = {
                'exit_code': -1,
                'duration': 0,
                'error': str(e),
                'path': category['path'],
                'description': category['description']
            }
    
    def run_existing_tests(self):
        """Run existing tests to ensure compatibility"""
        print(f"\n{'='*60}")
        print("RUNNING: Existing Tests (Compatibility Check)")
        print("Description: Verify existing tests still pass with OOP refactoring")
        print('='*60)
        
        try:
            # Run existing tests
            result = subprocess.run([
                sys.executable, '-m', 'pytest',
                'tests/test_factories.py',
                'tests/test_implementations.py',
                'tests/test_integration.py',
                '-v', '--tb=short'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            self.test_results['Existing Tests'] = {
                'exit_code': result.returncode,
                'duration': 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'description': 'Compatibility tests for existing functionality'
            }
            
            if result.returncode == 0:
                print("✅ Existing Tests - PASSED (Backward compatibility maintained)")
            else:
                print("❌ Existing Tests - FAILED (Backward compatibility issues)")
                
        except Exception as e:
            print(f"❌ Existing Tests - ERROR: {e}")
            self.test_results['Existing Tests'] = {
                'exit_code': -1,
                'duration': 0,
                'error': str(e),
                'description': 'Compatibility tests for existing functionality'
            }
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE TEST REPORT")
        print("="*80)
        
        total_duration = self.end_time - self.start_time
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get('exit_code') == 0)
        failed_tests = total_tests - passed_tests
        
        print(f"Total Duration: {total_duration:.2f}s")
        print(f"Total Test Categories: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print(f"\n{'='*60}")
        print("DETAILED RESULTS")
        print('='*60)
        
        for category_name, result in self.test_results.items():
            status = "✅ PASSED" if result.get('exit_code') == 0 else "❌ FAILED"
            duration = result.get('duration', 0)
            
            print(f"\n{category_name}")
            print(f"  Status: {status}")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Description: {result.get('description', 'N/A')}")
            
            if result.get('exit_code') != 0:
                print(f"  Error Details:")
                if 'stderr' in result and result['stderr']:
                    for line in result['stderr'].split('\n')[-5:]:
                        if line.strip():
                            print(f"    {line}")
                elif 'error' in result:
                    print(f"    {result['error']}")
        
        # Save detailed report
        self.save_detailed_report()
        
        print(f"\n{'='*80}")
        print("TEST SUITE COMPLETED")
        print("="*80)
        
        if failed_tests == 0:
            print("🎉 ALL TESTS PASSED! OOP refactoring is working correctly.")
        else:
            print(f"⚠️  {failed_tests} test categories failed. Please review the results above.")
        
        print(f"Report saved to: {self.project_root}/test_report.json")
    
    def save_detailed_report(self):
        """Save detailed test report to JSON file"""
        report = {
            'test_run_info': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.fromtimestamp(self.end_time).isoformat(),
                'total_duration': self.end_time - self.start_time,
                'project_root': str(self.project_root)
            },
            'summary': {
                'total_categories': len(self.test_results),
                'passed_categories': sum(1 for result in self.test_results.values() if result.get('exit_code') == 0),
                'failed_categories': sum(1 for result in self.test_results.values() if result.get('exit_code') != 0)
            },
            'detailed_results': self.test_results
        }
        
        report_file = self.project_root / 'test_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)


def main():
    """Main function to run the test suite"""
    runner = TestRunner()
    runner.run_all_tests()


if __name__ == "__main__":
    main()