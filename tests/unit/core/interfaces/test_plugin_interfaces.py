"""
插件接口全面单元测试
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List, Optional
import tempfile
import json
import os

from src.core.interfaces.plugin import (
    PluginInterface,
    PluginManager,
    PluginHookInterface,
    PluginMetadata,
    PluginStatus,
    plugin_hook
)
from src.core.exceptions import PluginError, PluginLoadError


class ConcretePlugin(PluginInterface):
    """用于测试的具体插件实现"""
    
    def __init__(self, name="test_plugin", version="1.0.0"):
        self._name = name
        self._version = version
        self._enabled = False
        self._initialized = False
        
    def get_name(self) -> str:
        return self._name
        
    def get_version(self) -> str:
        return self._version
        
    def initialize(self, config: Dict[str, Any] = None):
        """初始化插件"""
        if self._initialized:
            raise PluginError("Plugin already initialized")
        self._initialized = True
        self._config = config or {}
        
    def activate(self):
        """激活插件"""
        if not self._initialized:
            raise PluginError("Plugin not initialized")
        self._enabled = True
        
    def deactivate(self):
        """停用插件"""
        self._enabled = False
        
    def cleanup(self):
        """清理插件"""
        self._enabled = False
        self._initialized = False
        
    def is_enabled(self) -> bool:
        return self._enabled
        
    def get_config(self) -> Dict[str, Any]:
        return getattr(self, '_config', {})


class FailingPlugin(PluginInterface):
    """用于测试失败情况的插件实现"""
    
    def get_name(self) -> str:
        return "failing_plugin"
        
    def get_version(self) -> str:
        return "1.0.0"
        
    def initialize(self, config: Dict[str, Any] = None):
        raise Exception("Initialization failed")
        
    def activate(self):
        raise Exception("Activation failed")
        
    def deactivate(self):
        pass
        
    def cleanup(self):
        pass
        
    def is_enabled(self) -> bool:
        return False
        
    def get_config(self) -> Dict[str, Any]:
        return {}


class ConcretePluginHook(PluginHookInterface):
    """用于测试的具体插件钩子实现"""
    
    def __init__(self, hook_name="test_hook"):
        self._hook_name = hook_name
        self._callbacks = []
        
    def get_hook_name(self) -> str:
        return self._hook_name
        
    def register_callback(self, callback, priority: int = 0):
        """注册回调函数"""
        self._callbacks.append({'callback': callback, 'priority': priority})
        # 按优先级排序
        self._callbacks.sort(key=lambda x: x['priority'], reverse=True)
        
    def unregister_callback(self, callback):
        """取消注册回调函数"""
        self._callbacks = [cb for cb in self._callbacks if cb['callback'] != callback]
        
    def execute_callbacks(self, *args, **kwargs):
        """执行所有回调函数"""
        results = []
        for cb_info in self._callbacks:
            try:
                result = cb_info['callback'](*args, **kwargs)
                results.append(result)
            except Exception as e:
                results.append(e)
        return results


class TestPluginInterface(unittest.TestCase):
    
    def setUp(self):
        """测试前设置"""
        self.plugin = ConcretePlugin("test_plugin", "1.0.0")
        
    def test_plugin_creation(self):
        """测试插件创建"""
        self.assertEqual(self.plugin.get_name(), "test_plugin")
        self.assertEqual(self.plugin.get_version(), "1.0.0")
        self.assertFalse(self.plugin.is_enabled())
        
    def test_plugin_lifecycle_success(self):
        """测试插件生命周期成功流程"""
        # 初始化
        config = {'param1': 'value1'}
        self.plugin.initialize(config)
        self.assertEqual(self.plugin.get_config(), config)
        
        # 激活
        self.plugin.activate()
        self.assertTrue(self.plugin.is_enabled())
        
        # 停用
        self.plugin.deactivate()
        self.assertFalse(self.plugin.is_enabled())
        
        # 清理
        self.plugin.cleanup()
        self.assertFalse(self.plugin.is_enabled())
        
    def test_plugin_double_initialization(self):
        """测试插件重复初始化"""
        self.plugin.initialize()
        
        with self.assertRaises(PluginError):
            self.plugin.initialize()
            
    def test_plugin_activate_before_initialize(self):
        """测试未初始化就激活插件"""
        with self.assertRaises(PluginError):
            self.plugin.activate()
            
    def test_abstract_methods_enforcement(self):
        """测试抽象方法强制实现"""
        with self.assertRaises(TypeError):
            PluginInterface()


class TestPluginMetadata(unittest.TestCase):
    
    def test_plugin_metadata_creation(self):
        """测试插件元数据创建"""
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin",
            author="Test Author",
            dependencies=["dep1", "dep2"]
        )
        
        self.assertEqual(metadata.name, "test_plugin")
        self.assertEqual(metadata.version, "1.0.0")
        self.assertEqual(metadata.description, "Test plugin")
        self.assertEqual(metadata.author, "Test Author")
        self.assertEqual(metadata.dependencies, ["dep1", "dep2"])
        
    def test_plugin_metadata_defaults(self):
        """测试插件元数据默认值"""
        metadata = PluginMetadata(name="test", version="1.0")
        
        self.assertIsNone(metadata.description)
        self.assertIsNone(metadata.author)
        self.assertEqual(metadata.dependencies, [])
        
    def test_plugin_metadata_to_dict(self):
        """测试插件元数据转字典"""
        metadata = PluginMetadata(
            name="test",
            version="1.0",
            description="desc"
        )
        
        expected = {
            'name': 'test',
            'version': '1.0',
            'description': 'desc',
            'author': None,
            'dependencies': []
        }
        
        self.assertEqual(metadata.to_dict(), expected)


class TestPluginStatus(unittest.TestCase):
    
    def test_plugin_status_values(self):
        """测试插件状态枚举值"""
        expected_statuses = [
            "inactive", "active", "error", "disabled"
        ]
        
        actual_statuses = [status.value for status in PluginStatus]
        
        for expected in expected_statuses:
            self.assertIn(expected, actual_statuses)


class TestPluginManager(unittest.TestCase):
    
    def setUp(self):
        """测试前设置"""
        self.manager = PluginManager()
        self.plugin1 = ConcretePlugin("plugin1", "1.0.0")
        self.plugin2 = ConcretePlugin("plugin2", "2.0.0")
        
    def test_plugin_manager_creation(self):
        """测试插件管理器创建"""
        self.assertEqual(len(self.manager.get_all_plugins()), 0)
        
    def test_register_plugin_success(self):
        """测试插件注册成功"""
        self.manager.register_plugin("plugin1", self.plugin1)
        
        # 验证插件已注册
        self.assertEqual(len(self.manager.get_all_plugins()), 1)
        self.assertTrue(self.manager.is_plugin_registered("plugin1"))
        self.assertEqual(self.manager.get_plugin("plugin1"), self.plugin1)
        
    def test_register_duplicate_plugin(self):
        """测试注册重复插件"""
        self.manager.register_plugin("plugin1", self.plugin1)
        
        # 尝试注册同名插件应该抛出异常
        with self.assertRaises(PluginError):
            self.manager.register_plugin("plugin1", self.plugin2)
            
    def test_unregister_plugin_success(self):
        """测试插件注销成功"""
        self.manager.register_plugin("plugin1", self.plugin1)
        self.assertTrue(self.manager.is_plugin_registered("plugin1"))
        
        self.manager.unregister_plugin("plugin1")
        self.assertFalse(self.manager.is_plugin_registered("plugin1"))
        self.assertEqual(len(self.manager.get_all_plugins()), 0)
        
    def test_unregister_nonexistent_plugin(self):
        """测试注销不存在的插件"""
        with self.assertRaises(PluginError):
            self.manager.unregister_plugin("nonexistent")
            
    def test_get_nonexistent_plugin(self):
        """测试获取不存在的插件"""
        result = self.manager.get_plugin("nonexistent")
        self.assertIsNone(result)
        
    def test_activate_plugin_success(self):
        """测试激活插件成功"""
        self.manager.register_plugin("plugin1", self.plugin1)
        
        # 初始化插件
        self.manager.initialize_plugin("plugin1")
        
        # 激活插件
        self.manager.activate_plugin("plugin1")
        self.assertTrue(self.plugin1.is_enabled())
        
    def test_activate_plugin_failure(self):
        """测试激活插件失败"""
        failing_plugin = FailingPlugin()
        self.manager.register_plugin("failing", failing_plugin)
        
        # 激活应该失败
        with self.assertRaises(PluginError):
            self.manager.activate_plugin("failing")
            
    def test_deactivate_plugin_success(self):
        """测试停用插件成功"""
        self.manager.register_plugin("plugin1", self.plugin1)
        self.manager.initialize_plugin("plugin1")
        self.manager.activate_plugin("plugin1")
        
        # 停用插件
        self.manager.deactivate_plugin("plugin1")
        self.assertFalse(self.plugin1.is_enabled())
        
    def test_initialize_plugin_success(self):
        """测试初始化插件成功"""
        self.manager.register_plugin("plugin1", self.plugin1)
        
        config = {'test': 'value'}
        self.manager.initialize_plugin("plugin1", config)
        self.assertEqual(self.plugin1.get_config(), config)
        
    def test_initialize_plugin_failure(self):
        """测试初始化插件失败"""
        failing_plugin = FailingPlugin()
        self.manager.register_plugin("failing", failing_plugin)
        
        with self.assertRaises(PluginError):
            self.manager.initialize_plugin("failing")
            
    def test_cleanup_plugin_success(self):
        """测试清理插件成功"""
        self.manager.register_plugin("plugin1", self.plugin1)
        self.manager.initialize_plugin("plugin1")
        self.manager.activate_plugin("plugin1")
        
        self.manager.cleanup_plugin("plugin1")
        self.assertFalse(self.plugin1.is_enabled())
        
    def test_get_active_plugins(self):
        """测试获取活跃插件"""
        self.manager.register_plugin("plugin1", self.plugin1)
        self.manager.register_plugin("plugin2", self.plugin2)
        
        # 初始化和激活第一个插件
        self.manager.initialize_plugin("plugin1")
        self.manager.activate_plugin("plugin1")
        
        # 只初始化第二个插件
        self.manager.initialize_plugin("plugin2")
        
        active_plugins = self.manager.get_active_plugins()
        self.assertEqual(len(active_plugins), 1)
        self.assertEqual(active_plugins[0], self.plugin1)
        
    def test_cleanup_all_plugins(self):
        """测试清理所有插件"""
        self.manager.register_plugin("plugin1", self.plugin1)
        self.manager.register_plugin("plugin2", self.plugin2)
        
        self.manager.initialize_plugin("plugin1")
        self.manager.activate_plugin("plugin1")
        self.manager.initialize_plugin("plugin2")
        self.manager.activate_plugin("plugin2")
        
        self.manager.cleanup_all_plugins()
        
        self.assertFalse(self.plugin1.is_enabled())
        self.assertFalse(self.plugin2.is_enabled())


class TestPluginHookInterface(unittest.TestCase):
    
    def setUp(self):
        """测试前设置"""
        self.hook = ConcretePluginHook("test_hook")
        
    def test_hook_creation(self):
        """测试钩子创建"""
        self.assertEqual(self.hook.get_hook_name(), "test_hook")
        
    def test_register_callback_success(self):
        """测试注册回调成功"""
        def test_callback():
            return "callback_result"
            
        self.hook.register_callback(test_callback, priority=10)
        self.assertEqual(len(self.hook._callbacks), 1)
        self.assertEqual(self.hook._callbacks[0]['priority'], 10)
        
    def test_register_multiple_callbacks_with_priority(self):
        """测试注册多个带优先级的回调"""
        def callback1():
            return "result1"
            
        def callback2():
            return "result2"
            
        def callback3():
            return "result3"
            
        # 按不同优先级注册
        self.hook.register_callback(callback1, priority=5)
        self.hook.register_callback(callback2, priority=10)
        self.hook.register_callback(callback3, priority=1)
        
        # 验证按优先级排序
        self.assertEqual(len(self.hook._callbacks), 3)
        self.assertEqual(self.hook._callbacks[0]['priority'], 10)  # callback2
        self.assertEqual(self.hook._callbacks[1]['priority'], 5)   # callback1
        self.assertEqual(self.hook._callbacks[2]['priority'], 1)   # callback3
        
    def test_unregister_callback_success(self):
        """测试取消注册回调成功"""
        def test_callback():
            return "result"
            
        self.hook.register_callback(test_callback)
        self.assertEqual(len(self.hook._callbacks), 1)
        
        self.hook.unregister_callback(test_callback)
        self.assertEqual(len(self.hook._callbacks), 0)
        
    def test_execute_callbacks_success(self):
        """测试执行回调成功"""
        def callback1(x):
            return x * 2
            
        def callback2(x):
            return x + 10
            
        self.hook.register_callback(callback1)
        self.hook.register_callback(callback2)
        
        results = self.hook.execute_callbacks(5)
        
        self.assertEqual(len(results), 2)
        self.assertIn(10, results)  # 5 * 2
        self.assertIn(15, results)  # 5 + 10
        
    def test_execute_callbacks_with_exception(self):
        """测试执行回调遇到异常"""
        def good_callback():
            return "success"
            
        def bad_callback():
            raise Exception("Callback failed")
            
        self.hook.register_callback(good_callback)
        self.hook.register_callback(bad_callback)
        
        results = self.hook.execute_callbacks()
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], "success")
        self.assertIsInstance(results[1], Exception)
        
    def test_abstract_methods_enforcement(self):
        """测试抽象方法强制实现"""
        with self.assertRaises(TypeError):
            PluginHookInterface()


class TestPluginHookDecorator(unittest.TestCase):
    
    def test_plugin_hook_decorator(self):
        """测试插件钩子装饰器"""
        
        @plugin_hook("test_hook")
        def test_function():
            return "hooked"
            
        # 验证装饰器设置了属性
        self.assertEqual(test_function._hook_name, "test_hook")
        
        # 验证函数仍然可以正常调用
        result = test_function()
        self.assertEqual(result, "hooked")


class TestPluginErrorScenarios(unittest.TestCase):
    
    def test_plugin_error_instantiation(self):
        """测试插件错误实例化"""
        error = PluginError("Test plugin error")
        
        self.assertIsInstance(error, Exception)
        self.assertEqual(str(error), "Test plugin error")
        
    def test_plugin_load_error_instantiation(self):
        """测试插件加载错误实例化"""
        error = PluginLoadError("Failed to load plugin", plugin_name="test_plugin")
        
        self.assertIsInstance(error, PluginError)
        self.assertEqual(str(error), "Failed to load plugin")


class TestPluginManagerIntegration(unittest.TestCase):
    """插件管理器集成测试"""
    
    def test_full_plugin_lifecycle(self):
        """测试完整插件生命周期"""
        manager = PluginManager()
        plugin = ConcretePlugin("lifecycle_test", "1.0.0")
        
        # 注册
        manager.register_plugin("lifecycle_test", plugin)
        self.assertTrue(manager.is_plugin_registered("lifecycle_test"))
        
        # 初始化
        config = {'test_param': 'test_value'}
        manager.initialize_plugin("lifecycle_test", config)
        self.assertEqual(plugin.get_config(), config)
        
        # 激活
        manager.activate_plugin("lifecycle_test")
        self.assertTrue(plugin.is_enabled())
        
        # 验证在活跃插件列表中
        active_plugins = manager.get_active_plugins()
        self.assertIn(plugin, active_plugins)
        
        # 停用
        manager.deactivate_plugin("lifecycle_test")
        self.assertFalse(plugin.is_enabled())
        
        # 清理
        manager.cleanup_plugin("lifecycle_test")
        
        # 注销
        manager.unregister_plugin("lifecycle_test")
        self.assertFalse(manager.is_plugin_registered("lifecycle_test"))


if __name__ == '__main__':
    unittest.main()