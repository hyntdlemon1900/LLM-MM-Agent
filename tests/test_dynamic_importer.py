"""
动态导入器单元测试

演示工程化实现的优势：
- 可测试性
- 错误处理
- 缓存机制
- 路径管理
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / 'MMAgent'))

from utils.dynamic_importer import (
    DynamicImporter,
    get_default_importer,
    import_evaluator,
    get_evaluator_function
)


class TestDynamicImporter:
    """测试 DynamicImporter 类"""
    
    def test_initialization_with_default_base_path(self):
        """测试默认基础路径初始化"""
        importer = DynamicImporter()
        assert importer.base_path.exists()
        assert (importer.base_path / 'MMAgent').exists()
        assert (importer.base_path / 'MMBench').exists()
    
    def test_initialization_with_custom_base_path(self):
        """测试自定义基础路径"""
        custom_path = Path('/tmp/test_project')
        importer = DynamicImporter(base_path=custom_path)
        assert importer.base_path == custom_path.resolve()
    
    def test_get_runtime_path_success(self):
        """测试获取 runtime 路径（成功情况）"""
        importer = DynamicImporter()
        runtime_path = importer.get_runtime_path('LLMINA')
        
        assert runtime_path.exists()
        assert runtime_path.name == 'runtime'
        assert 'LLMINA' in str(runtime_path)
    
    def test_get_runtime_path_not_found(self):
        """测试获取不存在的 runtime 路径"""
        importer = DynamicImporter()
        
        with pytest.raises(FileNotFoundError) as exc_info:
            importer.get_runtime_path('NonExistentProblem')
        
        assert 'Runtime directory not found' in str(exc_info.value)
        assert 'NonExistentProblem' in str(exc_info.value)
    
    def test_get_runtime_path_caching(self):
        """测试 runtime 路径缓存"""
        importer = DynamicImporter()
        
        # 第一次调用
        path1 = importer.get_runtime_path('LLMINA')
        # 第二次调用应该使用缓存
        path2 = importer.get_runtime_path('LLMINA')
        
        assert path1 == path2
        # 验证缓存命中
        cache_info = importer.get_runtime_path.cache_info()
        assert cache_info.hits > 0
    
    def test_ensure_path_in_syspath(self):
        """测试路径添加到 sys.path"""
        importer = DynamicImporter()
        test_path = Path('/tmp/test_path')
        
        # 确保路径不在 sys.path 中
        if str(test_path) in sys.path:
            sys.path.remove(str(test_path))
        
        importer.ensure_path_in_syspath(test_path)
        assert str(test_path) in sys.path
        
        # 清理
        sys.path.remove(str(test_path))
    
    def test_import_evaluator_success(self):
        """测试成功导入评估器模块"""
        importer = DynamicImporter()
        
        try:
            module = importer.import_evaluator('LLMINA')
            
            # 验证模块属性
            assert hasattr(module, 'evaluate_solver_from_file')
            assert hasattr(module, 'SolverEvaluation')
            assert callable(module.evaluate_solver_from_file)
            
        except (FileNotFoundError, ImportError) as e:
            pytest.skip(f"Evaluation module not available: {e}")
    
    def test_import_evaluator_not_found(self):
        """测试导入不存在的评估器"""
        importer = DynamicImporter()
        
        with pytest.raises(FileNotFoundError) as exc_info:
            importer.import_evaluator('NonExistentProblem')
        
        assert 'Evaluation module not found' in str(exc_info.value)
    
    def test_get_evaluator_function_success(self):
        """测试获取评估器函数"""
        importer = DynamicImporter()
        
        try:
            func = importer.get_evaluator_function('evaluate_solver_from_file', 'LLMINA')
            
            assert callable(func)
            
        except (FileNotFoundError, ImportError) as e:
            pytest.skip(f"Evaluation module not available: {e}")
    
    def test_get_evaluator_function_not_found(self):
        """测试获取不存在的函数"""
        importer = DynamicImporter()
        
        try:
            with pytest.raises(AttributeError) as exc_info:
                importer.get_evaluator_function('non_existent_function', 'LLMINA')
            
            assert 'not found in evaluation module' in str(exc_info.value)
            
        except (FileNotFoundError, ImportError) as e:
            pytest.skip(f"Evaluation module not available: {e}")
    
    def test_module_caching(self):
        """测试模块缓存机制"""
        importer = DynamicImporter()
        
        try:
            # 第一次导入
            module1 = importer.import_evaluator('LLMINA')
            # 第二次导入应该使用缓存
            module2 = importer.import_evaluator('LLMINA')
            
            # 应该是同一个对象
            assert module1 is module2
            
        except (FileNotFoundError, ImportError) as e:
            pytest.skip(f"Evaluation module not available: {e}")
    
    def test_force_reload(self):
        """测试强制重新加载"""
        importer = DynamicImporter()
        
        try:
            # 第一次导入
            module1 = importer.import_evaluator('LLMINA')
            # 强制重新加载
            module2 = importer.import_evaluator('LLMINA', force_reload=True)
            
            # 功能相同但可能是不同的对象
            assert hasattr(module1, 'evaluate_solver_from_file')
            assert hasattr(module2, 'evaluate_solver_from_file')
            
        except (FileNotFoundError, ImportError) as e:
            pytest.skip(f"Evaluation module not available: {e}")
    
    def test_clear_cache(self):
        """测试清除缓存"""
        importer = DynamicImporter()
        
        try:
            # 导入模块
            importer.import_evaluator('LLMINA')
            assert len(importer._module_cache) > 0
            
            # 清除缓存
            importer.clear_cache()
            assert len(importer._module_cache) == 0
            assert importer.get_runtime_path.cache_info().hits == 0
            
        except (FileNotFoundError, ImportError) as e:
            pytest.skip(f"Evaluation module not available: {e}")


class TestConvenienceFunctions:
    """测试便捷函数"""
    
    def test_get_default_importer_singleton(self):
        """测试默认导入器是单例"""
        importer1 = get_default_importer()
        importer2 = get_default_importer()
        
        assert importer1 is importer2
    
    def test_import_evaluator_convenience_function(self):
        """测试便捷导入函数"""
        try:
            module = import_evaluator('LLMINA')
            assert hasattr(module, 'evaluate_solver_from_file')
            
        except (FileNotFoundError, ImportError) as e:
            pytest.skip(f"Evaluation module not available: {e}")
    
    def test_get_evaluator_function_convenience(self):
        """测试便捷获取函数"""
        try:
            func = get_evaluator_function('evaluate_solver_from_file', 'LLMINA')
            assert callable(func)
            
        except (FileNotFoundError, ImportError) as e:
            pytest.skip(f"Evaluation module not available: {e}")


class TestErrorHandling:
    """测试错误处理"""
    
    def test_runtime_path_validation(self):
        """测试 runtime 路径验证"""
        importer = DynamicImporter()
        
        with pytest.raises(FileNotFoundError) as exc_info:
            importer.get_runtime_path('InvalidProblem')
        
        error_msg = str(exc_info.value)
        assert 'Runtime directory not found' in error_msg
        assert 'InvalidProblem' in error_msg
        assert 'Please check' in error_msg
    
    def test_module_file_validation(self):
        """测试模块文件验证"""
        importer = DynamicImporter()
        
        with pytest.raises(FileNotFoundError) as exc_info:
            importer.import_evaluator('InvalidProblem')
        
        error_msg = str(exc_info.value)
        assert 'Evaluation module not found' in error_msg or 'Runtime directory not found' in error_msg
    
    def test_function_validation(self):
        """测试函数存在性验证"""
        importer = DynamicImporter()
        
        try:
            with pytest.raises(AttributeError) as exc_info:
                importer.get_evaluator_function('invalid_function', 'LLMINA')
            
            error_msg = str(exc_info.value)
            assert 'not found in evaluation module' in error_msg
            assert 'Available functions' in error_msg
            
        except (FileNotFoundError, ImportError) as e:
            pytest.skip(f"Evaluation module not available: {e}")


# 运行测试
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
