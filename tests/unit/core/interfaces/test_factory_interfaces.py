"""
工厂接口全面单元测试
"""

import unittest
from unittest.mock import Mock, MagicMock
from typing import Dict, Any

from src.core.interfaces.factory import (
    FactoryInterface,
    ModelFactoryInterface,
    DatasetFactoryInterface,
    ClientFactoryInterface,
    ServerFactoryInterface,
    AggregatorFactoryInterface,
    ComponentFactoryInterface,
    ComponentType,
    factory_register
)
from src.core.exceptions import FactoryError, ComponentCreationError


class ConcreteFactory(FactoryInterface[str]):
    """用于测试的具体工厂实现"""
    
    def __init__(self):
        self._registry = {}
        
    def create(self, component_type: str, config: Dict[str, Any], **kwargs) -> str:
        if component_type not in self._registry:
            raise ComponentCreationError(f"Component type '{component_type}' not registered")
        creator_class = self._registry[component_type]
        return creator_class.create(config, **kwargs)
        
    def register(self, component_type: str, creator_class):
        self._registry[component_type] = creator_class
        
    def unregister(self, component_type: str):
        self._registry.pop(component_type, None)
        
    def list_registered_types(self):
        return list(self._registry.keys())
        
    def is_registered(self, component_type: str) -> bool:
        return component_type in self._registry


class MockCreator:
    """Mock创建器类"""
    
    @staticmethod
    def create(config, **kwargs):
        return f"MockComponent(config={config})"


class ConcreteModelFactory(ModelFactoryInterface):
    """用于测试的具体模型工厂"""
    
    def __init__(self):
        self._models = {
            'mnist': ['mlp', 'vit'],
            'imdb': ['rnn', 'lstm', 'transformer']
        }
        
    def create(self, component_type: str, config: Dict[str, Any], **kwargs):
        return self.create_model(
            config.get('dataset'),
            component_type,
            config,
            **kwargs
        )
        
    def create_model(self, dataset: str, model_type: str, config: Dict[str, Any], **kwargs):
        if dataset not in self._models:
            raise ComponentCreationError(f"Dataset '{dataset}' not supported")
        if model_type not in self._models[dataset]:
            raise ComponentCreationError(f"Model '{model_type}' not supported for dataset '{dataset}'")
        return f"MockModel_{dataset}_{model_type}"
        
    def get_supported_datasets(self):
        return list(self._models.keys())
        
    def get_supported_models(self, dataset: str):
        return self._models.get(dataset, [])
        
    def register(self, component_type: str, creator_class):
        pass
        
    def unregister(self, component_type: str):
        pass
        
    def list_registered_types(self):
        return []
        
    def is_registered(self, component_type: str) -> bool:
        return False


class TestFactoryInterface(unittest.TestCase):
    
    def setUp(self):
        """测试前设置"""
        self.factory = ConcreteFactory()
        self.mock_creator = MockCreator()
        
    def test_factory_registration_and_creation(self):
        """测试工厂注册和创建"""
        # 注册组件创建器
        self.factory.register('test_component', self.mock_creator)
        
        # 验证注册成功
        self.assertTrue(self.factory.is_registered('test_component'))
        self.assertIn('test_component', self.factory.list_registered_types())
        
        # 创建组件
        config = {'param1': 'value1'}
        component = self.factory.create('test_component', config)
        
        self.assertEqual(component, "MockComponent(config={'param1': 'value1'})")
        
    def test_factory_unregistration(self):
        """测试工厂取消注册"""
        # 先注册
        self.factory.register('test_component', self.mock_creator)
        self.assertTrue(self.factory.is_registered('test_component'))
        
        # 取消注册
        self.factory.unregister('test_component')
        self.assertFalse(self.factory.is_registered('test_component'))
        self.assertEqual(len(self.factory.list_registered_types()), 0)
        
    def test_factory_creation_unregistered_component(self):
        """测试创建未注册组件失败"""
        with self.assertRaises(ComponentCreationError):
            self.factory.create('unregistered_component', {})
            
    def test_factory_multiple_registrations(self):
        """测试多个组件注册"""
        creators = {
            'component1': MockCreator(),
            'component2': MockCreator(),
            'component3': MockCreator()
        }
        
        # 注册多个组件
        for name, creator in creators.items():
            self.factory.register(name, creator)
            
        # 验证所有组件已注册
        registered_types = self.factory.list_registered_types()
        self.assertEqual(set(registered_types), set(creators.keys()))
        
        # 验证所有组件都能创建
        for name in creators.keys():
            component = self.factory.create(name, {})
            self.assertIn("MockComponent", component)


class TestModelFactoryInterface(unittest.TestCase):
    
    def setUp(self):
        """测试前设置"""
        self.factory = ConcreteModelFactory()
        
    def test_model_creation_success(self):
        """测试模型创建成功"""
        # MNIST MLP
        config = {'dataset': 'mnist'}
        model = self.factory.create_model('mnist', 'mlp', config)
        self.assertEqual(model, 'MockModel_mnist_mlp')
        
        # IMDB Transformer
        model = self.factory.create_model('imdb', 'transformer', config)
        self.assertEqual(model, 'MockModel_imdb_transformer')
        
    def test_model_creation_unsupported_dataset(self):
        """测试不支持的数据集"""
        config = {'dataset': 'unknown'}
        
        with self.assertRaises(ComponentCreationError):
            self.factory.create_model('unknown', 'mlp', config)
            
    def test_model_creation_unsupported_model(self):
        """测试不支持的模型"""
        config = {'dataset': 'mnist'}
        
        with self.assertRaises(ComponentCreationError):
            self.factory.create_model('mnist', 'transformer', config)  # transformer不支持MNIST
            
    def test_get_supported_datasets(self):
        """测试获取支持的数据集"""
        datasets = self.factory.get_supported_datasets()
        self.assertEqual(set(datasets), {'mnist', 'imdb'})
        
    def test_get_supported_models(self):
        """测试获取支持的模型"""
        # MNIST支持的模型
        mnist_models = self.factory.get_supported_models('mnist')
        self.assertEqual(set(mnist_models), {'mlp', 'vit'})
        
        # IMDB支持的模型
        imdb_models = self.factory.get_supported_models('imdb')
        self.assertEqual(set(imdb_models), {'rnn', 'lstm', 'transformer'})
        
        # 不存在的数据集
        unknown_models = self.factory.get_supported_models('unknown')
        self.assertEqual(len(unknown_models), 0)
        
    def test_create_via_base_interface(self):
        """测试通过基础接口创建"""
        config = {'dataset': 'imdb'}
        model = self.factory.create('lstm', config)
        self.assertEqual(model, 'MockModel_imdb_lstm')


class TestComponentType(unittest.TestCase):
    
    def test_component_type_values(self):
        """测试组件类型枚举值"""
        expected_types = [
            "client", "server", "aggregator", "model",
            "dataset", "strategy", "plugin"
        ]
        
        actual_types = [ct.value for ct in ComponentType]
        
        for expected in expected_types:
            self.assertIn(expected, actual_types)


class TestFactoryRegisterDecorator(unittest.TestCase):
    
    def test_factory_register_decorator(self):
        """测试工厂注册装饰器"""
        
        @factory_register(ComponentType.MODEL, "test_model")
        class TestModelCreator:
            def create(self, config):
                return f"TestModel(config={config})"
                
        # 验证装饰器设置了属性
        self.assertEqual(TestModelCreator._factory_type, ComponentType.MODEL)
        self.assertEqual(TestModelCreator._component_name, "test_model")


class TestAbstractInterfaceEnforcement(unittest.TestCase):
    
    def test_factory_interface_abstract_methods(self):
        """测试工厂接口抽象方法强制实现"""
        
        # 尝试直接实例化抽象类应该失败
        with self.assertRaises(TypeError):
            FactoryInterface()
            
    def test_model_factory_interface_abstract_methods(self):
        """测试模型工厂接口抽象方法强制实现"""
        
        with self.assertRaises(TypeError):
            ModelFactoryInterface()
            
    def test_dataset_factory_interface_abstract_methods(self):
        """测试数据集工厂接口抽象方法强制实现"""
        
        with self.assertRaises(TypeError):
            DatasetFactoryInterface()
            
    def test_client_factory_interface_abstract_methods(self):
        """测试客户端工厂接口抽象方法强制实现"""
        
        with self.assertRaises(TypeError):
            ClientFactoryInterface()
            
    def test_server_factory_interface_abstract_methods(self):
        """测试服务器工厂接口抽象方法强制实现"""
        
        with self.assertRaises(TypeError):
            ServerFactoryInterface()
            
    def test_aggregator_factory_interface_abstract_methods(self):
        """测试聚合器工厂接口抽象方法强制实现"""
        
        with self.assertRaises(TypeError):
            AggregatorFactoryInterface()
            
    def test_component_factory_interface_abstract_methods(self):
        """测试组件工厂接口抽象方法强制实现"""
        
        with self.assertRaises(TypeError):
            ComponentFactoryInterface()


class TestFactoryErrorScenarios(unittest.TestCase):
    
    def setUp(self):
        """测试前设置"""
        self.factory = ConcreteFactory()
        
    def test_factory_error_propagation(self):
        """测试工厂错误传播"""
        
        class FailingCreator:
            @staticmethod
            def create(config, **kwargs):
                raise Exception("Creation failed")
                
        self.factory.register('failing_component', FailingCreator())
        
        # 验证创建错误被正确传播
        with self.assertRaises(Exception):
            self.factory.create('failing_component', {})
            
    def test_duplicate_registration_handling(self):
        """测试重复注册处理"""
        creator1 = MockCreator()
        creator2 = MockCreator()
        
        # 首次注册
        self.factory.register('component', creator1)
        self.assertEqual(self.factory._registry['component'], creator1)
        
        # 重复注册应该覆盖
        self.factory.register('component', creator2)
        self.assertEqual(self.factory._registry['component'], creator2)
        
    def test_unregister_nonexistent_component(self):
        """测试取消注册不存在的组件"""
        # 不应该抛出异常
        self.factory.unregister('nonexistent')
        
        # 验证状态没有改变
        self.assertEqual(len(self.factory.list_registered_types()), 0)


if __name__ == '__main__':
    unittest.main()