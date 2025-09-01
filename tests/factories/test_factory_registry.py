"""
工厂注册系统测试

测试FactoryRegistry的所有功能，包括：
- 工厂注册和注销
- 组件创建器注册和管理
- 线程安全性
- 单例模式
- 统计信息和验证
"""

import unittest
import threading
import time
from unittest.mock import Mock

from src.core.exceptions import FactoryError
from src.core.interfaces.factory import FactoryInterface, ComponentType
from src.factories.factory_registry import (
    FactoryRegistry, get_factory_registry,
    register_factory, register_component_creator
)


class TestFactoryInterface(FactoryInterface):
    """测试用工厂接口实现"""
    
    def create(self, component_type, config, **kwargs):
        return f"mock_component_{component_type}"
    
    def register(self, component_type, creator_class):
        pass
    
    def unregister(self, component_type):
        pass
    
    def list_registered_types(self):
        return ["test_type"]
    
    def is_registered(self, component_type):
        return component_type == "test_type"


class MockComponentCreator:
    """模拟组件创建器"""
    
    @staticmethod
    def create(config):
        return "mock_component"


class TestFactoryRegistry(unittest.TestCase):
    """工厂注册系统测试"""
    
    def setUp(self):
        """每个测试前的设置"""
        self.registry = FactoryRegistry()
        # 清空注册表，确保测试隔离
        self.registry.clear_registry()
    
    def tearDown(self):
        """每个测试后的清理"""
        self.registry.clear_registry()
    
    def test_singleton_pattern(self):
        """测试单例模式"""
        registry1 = FactoryRegistry()
        registry2 = FactoryRegistry()
        
        # 应该是同一个实例
        self.assertIs(registry1, registry2)
        
        # 全局函数返回的也应该是同一个实例
        global_registry = get_factory_registry()
        self.assertIs(registry1, global_registry)
    
    def test_factory_registration(self):
        """测试工厂注册功能"""
        factory_type = "test_factory"
        factory_class = TestFactoryInterface
        
        # 注册工厂
        self.registry.register_factory(factory_type, factory_class)
        
        # 验证注册成功
        self.assertTrue(self.registry.is_factory_registered(factory_type))
        self.assertIn(factory_type, self.registry.list_registered_factories())
        
        # 获取工厂实例
        factory_instance = self.registry.get_factory(factory_type)
        self.assertIsInstance(factory_instance, TestFactoryInterface)
        
        # 再次获取应该返回同一个实例（单例）
        factory_instance2 = self.registry.get_factory(factory_type)
        self.assertIs(factory_instance, factory_instance2)
    
    def test_factory_registration_duplicate(self):
        """测试重复注册工厂"""
        factory_type = "test_factory"
        factory_class = TestFactoryInterface
        
        # 首次注册
        self.registry.register_factory(factory_type, factory_class)
        
        # 重复注册应该抛出异常
        with self.assertRaises(FactoryError):
            self.registry.register_factory(factory_type, factory_class)
    
    def test_factory_registration_invalid_class(self):
        """测试注册无效工厂类"""
        factory_type = "invalid_factory"
        
        # 注册不是FactoryInterface子类的类
        class InvalidFactory:
            pass
        
        with self.assertRaises(FactoryError):
            self.registry.register_factory(factory_type, InvalidFactory)
    
    def test_factory_unregistration(self):
        """测试工厂注销功能"""
        factory_type = "test_factory"
        factory_class = TestFactoryInterface
        
        # 注册工厂
        self.registry.register_factory(factory_type, factory_class)
        self.assertTrue(self.registry.is_factory_registered(factory_type))
        
        # 注销工厂
        self.registry.unregister_factory(factory_type)
        self.assertFalse(self.registry.is_factory_registered(factory_type))
        self.assertNotIn(factory_type, self.registry.list_registered_factories())
    
    def test_get_unknown_factory(self):
        """测试获取未注册的工厂"""
        with self.assertRaises(FactoryError):
            self.registry.get_factory("unknown_factory")
    
    def test_component_creator_registration(self):
        """测试组件创建器注册"""
        component_type = ComponentType.MODEL
        creator_name = "test_creator"
        creator_class = MockComponentCreator
        
        # 注册创建器
        self.registry.register_component_creator(component_type, creator_name, creator_class)
        
        # 验证注册成功
        self.assertTrue(self.registry.is_component_creator_registered(component_type, creator_name))
        self.assertIn(creator_name, self.registry.list_component_creators(component_type))
        
        # 获取创建器
        retrieved_creator = self.registry.get_component_creator(component_type, creator_name)
        self.assertIs(retrieved_creator, creator_class)
    
    def test_component_creator_duplicate_registration(self):
        """测试重复注册组件创建器"""
        component_type = ComponentType.MODEL
        creator_name = "test_creator"
        creator_class = MockComponentCreator
        
        # 首次注册
        self.registry.register_component_creator(component_type, creator_name, creator_class)
        
        # 重复注册应该抛出异常
        with self.assertRaises(FactoryError):
            self.registry.register_component_creator(component_type, creator_name, creator_class)
    
    def test_component_creator_unregistration(self):
        """测试组件创建器注销"""
        component_type = ComponentType.MODEL
        creator_name = "test_creator"
        creator_class = MockComponentCreator
        
        # 注册创建器
        self.registry.register_component_creator(component_type, creator_name, creator_class)
        self.assertTrue(self.registry.is_component_creator_registered(component_type, creator_name))
        
        # 注销创建器
        self.registry.unregister_component_creator(component_type, creator_name)
        self.assertFalse(self.registry.is_component_creator_registered(component_type, creator_name))
        self.assertNotIn(creator_name, self.registry.list_component_creators(component_type))
    
    def test_get_unknown_component_creator(self):
        """测试获取未注册的组件创建器"""
        result = self.registry.get_component_creator(ComponentType.MODEL, "unknown_creator")
        self.assertIsNone(result)
    
    def test_clear_registry(self):
        """测试清空注册表"""
        # 注册一些数据
        self.registry.register_factory("test_factory", TestFactoryInterface)
        self.registry.register_component_creator(ComponentType.MODEL, "test_creator", MockComponentCreator)
        
        # 验证数据存在
        self.assertTrue(self.registry.is_factory_registered("test_factory"))
        self.assertTrue(self.registry.is_component_creator_registered(ComponentType.MODEL, "test_creator"))
        
        # 清空注册表
        self.registry.clear_registry()
        
        # 验证数据被清空
        self.assertFalse(self.registry.is_factory_registered("test_factory"))
        self.assertFalse(self.registry.is_component_creator_registered(ComponentType.MODEL, "test_creator"))
        self.assertEqual(len(self.registry.list_registered_factories()), 0)
    
    def test_registry_stats(self):
        """测试注册统计信息"""
        # 获取初始统计
        initial_stats = self.registry.get_registry_stats()
        self.assertIsInstance(initial_stats, dict)
        
        # 注册一些数据
        self.registry.register_factory("test_factory", TestFactoryInterface)
        self.registry.register_component_creator(ComponentType.MODEL, "model_creator", MockComponentCreator)
        self.registry.register_component_creator(ComponentType.DATASET, "dataset_creator", MockComponentCreator)
        
        # 获取工厂实例（这会增加实例计数）
        self.registry.get_factory("test_factory")
        
        # 检查统计信息
        stats = self.registry.get_registry_stats()
        self.assertEqual(stats['factory_types'], 1)
        self.assertEqual(stats['factory_instances'], 1)
        self.assertEqual(stats['component_creators']['model'], 1)
        self.assertEqual(stats['component_creators']['dataset'], 1)
        self.assertEqual(stats['total_creators'], 2)
        self.assertIn("test_factory", stats['registered_factories'])
    
    def test_thread_safety(self):
        """测试线程安全性"""
        results = []
        errors = []
        
        def register_factory_worker(worker_id):
            try:
                factory_type = f"test_factory_{worker_id}"
                self.registry.register_factory(factory_type, TestFactoryInterface)
                results.append(factory_type)
            except Exception as e:
                errors.append(e)
        
        def register_component_worker(worker_id):
            try:
                creator_name = f"creator_{worker_id}"
                self.registry.register_component_creator(
                    ComponentType.MODEL, creator_name, MockComponentCreator
                )
                results.append(creator_name)
            except Exception as e:
                errors.append(e)
        
        # 创建多个线程同时操作
        threads = []
        for i in range(10):
            t1 = threading.Thread(target=register_factory_worker, args=(i,))
            t2 = threading.Thread(target=register_component_worker, args=(i,))
            threads.extend([t1, t2])
        
        # 启动所有线程
        for thread in threads:
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
        self.assertEqual(len(results), 20)  # 10个工厂 + 10个组件创建器
        
        # 验证所有注册都成功
        self.assertEqual(len(self.registry.list_registered_factories()), 10)
        self.assertEqual(len(self.registry.list_component_creators(ComponentType.MODEL)), 10)
    
    def test_decorator_register_factory(self):
        """测试工厂注册装饰器"""
        @register_factory("decorated_factory")
        class DecoratedFactory(FactoryInterface):
            def create(self, component_type, config, **kwargs):
                return "decorated"
            
            def register(self, component_type, creator_class):
                pass
                
            def unregister(self, component_type):
                pass
                
            def list_registered_types(self):
                return []
                
            def is_registered(self, component_type):
                return False
        
        # 验证装饰器自动注册了工厂
        self.assertTrue(self.registry.is_factory_registered("decorated_factory"))
        
        # 验证类属性被正确设置
        self.assertEqual(DecoratedFactory._factory_type, "decorated_factory")
    
    def test_decorator_register_component_creator(self):
        """测试组件创建器注册装饰器"""
        @register_component_creator(ComponentType.CLIENT, "decorated_creator")
        class DecoratedCreator:
            @staticmethod
            def create(config):
                return "decorated_component"
        
        # 验证装饰器自动注册了创建器
        self.assertTrue(
            self.registry.is_component_creator_registered(ComponentType.CLIENT, "decorated_creator")
        )
        
        # 验证类属性被正确设置
        self.assertEqual(DecoratedCreator._component_type, ComponentType.CLIENT)
        self.assertEqual(DecoratedCreator._component_name, "decorated_creator")
    
    def test_factory_creation_with_kwargs(self):
        """测试带参数的工厂创建"""
        class ParameterizedFactory(FactoryInterface):
            def __init__(self, param1=None, param2=None):
                self.param1 = param1
                self.param2 = param2
            
            def create(self, component_type, config, **kwargs):
                return f"param_{self.param1}_{self.param2}"
            
            def register(self, component_type, creator_class):
                pass
                
            def unregister(self, component_type):
                pass
                
            def list_registered_types(self):
                return []
                
            def is_registered(self, component_type):
                return False
        
        # 注册工厂
        self.registry.register_factory("param_factory", ParameterizedFactory)
        
        # 使用参数创建工厂实例
        factory = self.registry.get_factory("param_factory", param1="test", param2="value")
        
        self.assertEqual(factory.param1, "test")
        self.assertEqual(factory.param2, "value")
        self.assertEqual(factory.create("test", {}), "param_test_value")


if __name__ == '__main__':
    unittest.main()