"""
FederatedComponent 基类测试
"""

import unittest
import logging
from datetime import datetime
from unittest.mock import Mock, patch

from src.core.base.component import FederatedComponent, ComponentStatus, ComponentRegistry
from src.core.exceptions import ConfigValidationError


class TestFederatedComponentImplImpl(FederatedComponent):
    """用于测试的具体实现类"""
    
    def _validate_config(self):
        # 简单的配置验证
        if 'required_param' in self._config and self._config['required_param'] is None:
            raise ConfigValidationError("required_param cannot be None")
    
    def _initialize(self):
        self._set_status(ComponentStatus.READY)


class TestFederatedComponentImplClass(unittest.TestCase):
    
    def setUp(self):
        """测试前设置"""
        ComponentRegistry.clear()
        
    def tearDown(self):
        """测试后清理"""
        ComponentRegistry.clear()
        
    def test_component_creation(self):
        """测试组件创建"""
        component = TestFederatedComponentImplImpl()
        
        # 验证基本属性
        self.assertIsNotNone(component.component_id)
        self.assertEqual(component.status, ComponentStatus.READY)
        self.assertIsInstance(component.created_at, datetime)
        self.assertIsInstance(component.logger, logging.Logger)
        
    def test_component_with_custom_id(self):
        """测试使用自定义ID创建组件"""
        custom_id = "test_component_123"
        component = TestFederatedComponentImplImpl(component_id=custom_id)
        
        self.assertEqual(component.component_id, custom_id)
        
    def test_component_with_config(self):
        """测试使用配置创建组件"""
        config = {"param1": "value1", "param2": 42}
        component = TestFederatedComponentImplImpl(config=config)
        
        self.assertEqual(component.get_config_value("param1"), "value1")
        self.assertEqual(component.get_config_value("param2"), 42)
        self.assertEqual(component.get_config_value("non_existent", "default"), "default")
        
    def test_nested_config_access(self):
        """测试嵌套配置访问"""
        config = {
            "model": {
                "hidden_size": 128,
                "layers": {
                    "depth": 4
                }
            }
        }
        component = TestFederatedComponentImplImpl(config=config)
        
        self.assertEqual(component.get_config_value("model.hidden_size"), 128)
        self.assertEqual(component.get_config_value("model.layers.depth"), 4)
        self.assertIsNone(component.get_config_value("model.non_existent"))
        
    def test_metadata_management(self):
        """测试元数据管理"""
        component = TestFederatedComponentImplImpl()
        
        # 设置元数据
        component.set_metadata("test_key", "test_value")
        self.assertEqual(component.get_metadata("test_key"), "test_value")
        
        # 获取不存在的元数据
        self.assertIsNone(component.get_metadata("non_existent"))
        self.assertEqual(component.get_metadata("non_existent", "default"), "default")
        
    def test_config_validation_success(self):
        """测试配置验证成功"""
        config = {"required_param": "valid_value"}
        component = TestFederatedComponentImplImpl(config=config)
        self.assertEqual(component.status, ComponentStatus.READY)
        
    def test_config_validation_failure(self):
        """测试配置验证失败"""
        config = {"required_param": None}
        with self.assertRaises(ConfigValidationError):
            TestFederatedComponentImplImpl(config=config)
            
    def test_config_update_success(self):
        """测试配置更新成功"""
        component = TestFederatedComponentImplImpl()
        new_config = {"new_param": "new_value"}
        
        component.update_config(new_config)
        self.assertEqual(component.get_config_value("new_param"), "new_value")
        
    def test_config_update_failure_rollback(self):
        """测试配置更新失败时回滚"""
        component = TestFederatedComponentImplImpl(config={"required_param": "valid"})
        original_config = component.config
        
        # 尝试更新为无效配置
        with self.assertRaises(ConfigValidationError):
            component.update_config({"required_param": None})
            
        # 验证配置已回滚
        self.assertEqual(component.config, original_config)
        
    def test_string_representation(self):
        """测试字符串表示"""
        component = TestFederatedComponentImplImpl(component_id="test_123")
        
        str_repr = str(component)
        self.assertIn("TestFederatedComponentImpl", str_repr)
        self.assertIn("test_123", str_repr)
        self.assertIn(ComponentStatus.READY.value, str_repr)
        
        repr_str = repr(component)
        self.assertIn("TestFederatedComponentImpl", repr_str)
        self.assertIn("test_123", repr_str)


class TestComponentStatus(unittest.TestCase):
    
    def test_status_values(self):
        """测试状态值"""
        expected_statuses = [
            "created", "initializing", "ready", "running", 
            "paused", "error", "stopped"
        ]
        
        actual_statuses = [status.value for status in ComponentStatus.all_statuses()]
        self.assertEqual(set(actual_statuses), set(expected_statuses))


class TestComponentRegistry(unittest.TestCase):
    
    def setUp(self):
        """测试前设置"""
        ComponentRegistry.clear()
        
    def tearDown(self):
        """测试后清理"""
        ComponentRegistry.clear()
        
    def test_register_and_get_component(self):
        """测试注册和获取组件"""
        component = TestFederatedComponentImplImpl(component_id="test_component")
        ComponentRegistry.register(component)
        
        retrieved = ComponentRegistry.get_component("test_component")
        self.assertEqual(component, retrieved)
        
    def test_get_nonexistent_component(self):
        """测试获取不存在的组件"""
        result = ComponentRegistry.get_component("non_existent")
        self.assertIsNone(result)
        
    def test_unregister_component(self):
        """测试取消注册组件"""
        component = TestFederatedComponentImplImpl(component_id="test_component")
        ComponentRegistry.register(component)
        
        ComponentRegistry.unregister("test_component")
        result = ComponentRegistry.get_component("test_component")
        self.assertIsNone(result)
        
    def test_list_components(self):
        """测试列出组件"""
        component1 = TestFederatedComponentImplImpl(component_id="component1")
        component2 = TestFederatedComponentImplImpl(component_id="component2")
        
        ComponentRegistry.register(component1)
        ComponentRegistry.register(component2)
        
        components = ComponentRegistry.list_components()
        self.assertEqual(len(components), 2)
        self.assertIn(component1, components)
        self.assertIn(component2, components)
        
    def test_list_components_by_type(self):
        """测试按类型列出组件"""
        component1 = TestFederatedComponentImplImpl(component_id="component1")
        component2 = TestFederatedComponentImplImpl(component_id="component2")
        
        ComponentRegistry.register(component1)
        ComponentRegistry.register(component2)
        
        # 测试按类型过滤
        components = ComponentRegistry.list_components(TestFederatedComponentImplImpl)
        self.assertEqual(len(components), 2)
        
        # 测试不匹配的类型
        components = ComponentRegistry.list_components(str)  # 不匹配的类型
        self.assertEqual(len(components), 0)
        
    def test_clear_registry(self):
        """测试清空注册表"""
        component = TestFederatedComponentImplImpl(component_id="test_component")
        ComponentRegistry.register(component)
        
        ComponentRegistry.clear()
        components = ComponentRegistry.list_components()
        self.assertEqual(len(components), 0)


if __name__ == '__main__':
    unittest.main()