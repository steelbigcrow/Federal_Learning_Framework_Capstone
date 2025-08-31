"""
测试运行器和配置
"""

import unittest
import sys
import os

# 添加src路径到sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def run_core_tests():
    """运行核心模块的所有测试"""
    
    # 发现并运行测试
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 返回测试结果
    return result.wasSuccessful()

def run_specific_test_module(module_name):
    """运行特定的测试模块"""
    
    try:
        # 导入测试模块
        test_module = __import__(f'tests.core.{module_name}', fromlist=[''])
        
        # 创建测试套件
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_module)
        
        # 运行测试
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
        
    except ImportError as e:
        print(f"无法导入测试模块 {module_name}: {e}")
        return False

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='运行核心模块测试')
    parser.add_argument('--module', '-m', type=str, help='运行特定的测试模块')
    parser.add_argument('--list', '-l', action='store_true', help='列出所有可用的测试模块')
    
    args = parser.parse_args()
    
    if args.list:
        print("可用的测试模块:")
        for root, dirs, files in os.walk(os.path.dirname(__file__)):
            for file in files:
                if file.startswith('test_') and file.endswith('.py'):
                    module_path = os.path.relpath(
                        os.path.join(root, file),
                        os.path.dirname(__file__)
                    )
                    module_name = module_path.replace(os.sep, '.').replace('.py', '')
                    print(f"  {module_name}")
    elif args.module:
        success = run_specific_test_module(args.module)
        sys.exit(0 if success else 1)
    else:
        success = run_core_tests()
        sys.exit(0 if success else 1)