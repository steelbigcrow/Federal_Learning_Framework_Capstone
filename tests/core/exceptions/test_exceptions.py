"""
异常体系测试
"""

import unittest
from unittest.mock import Mock

from src.core.exceptions import (
    FederatedLearningException,
    ErrorSeverity,
    ErrorCategory,
    ConfigurationError,
    ClientError,
    ServerError,
    AggregationError,
    ExceptionHandler
)


class TestFederatedLearningException(unittest.TestCase):
    
    def test_basic_exception_creation(self):
        """测试基础异常创建"""
        exc = FederatedLearningException("Test error message")
        
        self.assertEqual(exc.message, "Test error message")
        self.assertEqual(exc.severity, ErrorSeverity.MEDIUM)
        self.assertEqual(exc.category, ErrorCategory.SYSTEM)
        self.assertIsNotNone(exc.error_code)
        self.assertEqual(exc.context, {})
        self.assertIsNone(exc.cause)
        
    def test_exception_with_all_parameters(self):
        """测试包含所有参数的异常创建"""
        context = {"client_id": 1, "round": 5}
        cause = ValueError("Original error")
        
        exc = FederatedLearningException(
            message="Test error",
            error_code="TEST_001",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.TRAINING,
            context=context,
            cause=cause
        )
        
        self.assertEqual(exc.message, "Test error")
        self.assertEqual(exc.error_code, "TEST_001")
        self.assertEqual(exc.severity, ErrorSeverity.HIGH)
        self.assertEqual(exc.category, ErrorCategory.TRAINING)
        self.assertEqual(exc.context, context)
        self.assertEqual(exc.cause, cause)
        
    def test_error_code_generation(self):
        """测试错误代码自动生成"""
        exc = FederatedLearningException("Test error")
        self.assertEqual(exc.error_code, "SYSTEM_FEDERATEDLEARNINGEXCEPTION")
        
    def test_full_message(self):
        """测试完整错误消息"""
        context = {"client_id": 1}
        cause = ValueError("Original error")
        
        exc = FederatedLearningException(
            message="Test error",
            error_code="TEST_001",
            context=context,
            cause=cause
        )
        
        full_message = exc.get_full_message()
        self.assertIn("[TEST_001]", full_message)
        self.assertIn("Test error", full_message)
        self.assertIn("client_id=1", full_message)
        self.assertIn("Caused by:", full_message)
        
    def test_to_dict(self):
        """测试转换为字典"""
        context = {"test": "value"}
        cause = ValueError("Original")
        
        exc = FederatedLearningException(
            message="Test error",
            error_code="TEST_001",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.DATA,
            context=context,
            cause=cause
        )
        
        exc_dict = exc.to_dict()
        
        self.assertEqual(exc_dict["error_code"], "TEST_001")
        self.assertEqual(exc_dict["message"], "Test error")
        self.assertEqual(exc_dict["severity"], "high")
        self.assertEqual(exc_dict["category"], "data")
        self.assertEqual(exc_dict["context"], context)
        self.assertEqual(exc_dict["cause"], "Original")
        self.assertIsNotNone(exc_dict["traceback"])
        
    def test_string_representation(self):
        """测试字符串表示"""
        exc = FederatedLearningException("Test error", error_code="TEST_001")
        str_repr = str(exc)
        self.assertIn("[TEST_001]", str_repr)
        self.assertIn("Test error", str_repr)


class TestSpecificExceptions(unittest.TestCase):
    
    def test_configuration_error(self):
        """测试配置错误"""
        exc = ConfigurationError("Invalid config")
        
        self.assertEqual(exc.severity, ErrorSeverity.HIGH)
        self.assertEqual(exc.category, ErrorCategory.CONFIGURATION)
        self.assertIn("CONFIGURATION", exc.error_code)
        
    def test_client_error_with_client_id(self):
        """测试带客户端ID的客户端错误"""
        exc = ClientError("Client failed", client_id=5)
        
        self.assertEqual(exc.category, ErrorCategory.TRAINING)
        self.assertEqual(exc.context.get("client_id"), 5)
        
    def test_server_error(self):
        """测试服务器错误"""
        exc = ServerError("Server failed")
        
        self.assertEqual(exc.category, ErrorCategory.SYSTEM)
        self.assertIn("SERVER", exc.error_code)
        
    def test_aggregation_error(self):
        """测试聚合错误"""
        exc = AggregationError("Aggregation failed")
        
        self.assertEqual(exc.category, ErrorCategory.AGGREGATION)
        self.assertIn("AGGREGATION", exc.error_code)


class TestErrorEnums(unittest.TestCase):
    
    def test_error_severity_values(self):
        """测试错误严重程度枚举值"""
        expected_values = ["low", "medium", "high", "critical"]
        actual_values = [severity.value for severity in ErrorSeverity]
        
        self.assertEqual(set(actual_values), set(expected_values))
        
    def test_error_category_values(self):
        """测试错误类别枚举值"""
        expected_categories = [
            "configuration", "network", "data", "model", "training",
            "aggregation", "plugin", "security", "resource", "system"
        ]
        actual_categories = [category.value for category in ErrorCategory]
        
        for expected in expected_categories:
            self.assertIn(expected, actual_categories)


class TestExceptionHandler(unittest.TestCase):
    
    def test_handle_federated_exception_reraise(self):
        """测试处理联邦学习异常并重新抛出"""
        exc = FederatedLearningException("Test error")
        mock_logger = Mock()
        
        with self.assertRaises(FederatedLearningException):
            ExceptionHandler.handle_exception(exc, logger=mock_logger, reraise=True)
            
        # 验证日志被调用
        mock_logger.error.assert_called_once()
        
    def test_handle_federated_exception_no_reraise(self):
        """测试处理联邦学习异常但不重新抛出"""
        exc = FederatedLearningException("Test error", error_code="TEST_001")
        mock_logger = Mock()
        
        result = ExceptionHandler.handle_exception(exc, logger=mock_logger, reraise=False)
        
        self.assertIsNotNone(result)
        self.assertEqual(result["error_code"], "TEST_001")
        self.assertEqual(result["message"], "Test error")
        mock_logger.error.assert_called_once()
        
    def test_handle_standard_exception(self):
        """测试处理标准Python异常"""
        exc = ValueError("Standard error")
        context = {"test": "context"}
        
        result = ExceptionHandler.handle_exception(
            exc, reraise=False, context=context
        )
        
        self.assertIsNotNone(result)
        self.assertIn("FEDERATEDLEARNINGEXCEPTION", result["error_code"])
        self.assertEqual(result["message"], "Standard error")
        self.assertEqual(result["context"], context)
        
    def test_create_error_response_federated_exception(self):
        """测试为联邦学习异常创建错误响应"""
        exc = FederatedLearningException("Test error", error_code="TEST_001")
        
        response = ExceptionHandler.create_error_response(exc)
        
        self.assertFalse(response["success"])
        self.assertEqual(response["error"]["error_code"], "TEST_001")
        self.assertEqual(response["error"]["message"], "Test error")
        
    def test_create_error_response_standard_exception(self):
        """测试为标准异常创建错误响应"""
        exc = ValueError("Standard error")
        
        response = ExceptionHandler.create_error_response(exc)
        
        self.assertFalse(response["success"])
        self.assertEqual(response["error"]["error_code"], "UNKNOWN_ERROR")
        self.assertEqual(response["error"]["message"], "Standard error")
        self.assertEqual(response["error"]["severity"], "medium")
        self.assertEqual(response["error"]["category"], "system")


if __name__ == '__main__':
    unittest.main()