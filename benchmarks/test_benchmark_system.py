#!/usr/bin/env python3
"""
Quick Test Script - Verify Benchmark System
===========================================

This script performs a quick test of the benchmark system to ensure
all components are working correctly before running the full comparison.

Date: 2025
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        from benchmarks.benchmark_models import *
        print("✓ benchmark_models imported successfully")
    except Exception as e:
        print(f"✗ Error importing benchmark_models: {e}")
        return False
    
    try:
        from benchmarks.model_factory import ModelFactory, ModelComparator
        print("✓ model_factory imported successfully")
    except Exception as e:
        print(f"✗ Error importing model_factory: {e}")
        return False
    
    try:
        from benchmarks.comprehensive_evaluator import ComprehensiveModelEvaluator
        print("✓ comprehensive_evaluator imported successfully")
    except Exception as e:
        print(f"✗ Error importing comprehensive_evaluator: {e}")
        return False
    
    return True

def test_model_creation():
    """Test model creation"""
    print("\nTesting model creation...")
    
    try:
        from model_factory import ModelFactory
        
        factory = ModelFactory()
        
        # Test creating different types of models
        test_models = [
            'ma_cross_5_20',  # Technical
            'random_forest_100',  # ML
            'logistic_regression'  # ML
        ]
        
        for model_name in test_models:
            model = factory.create_model(model_name, '1d')
            if model is not None:
                print(f"✓ Created {model_name}")
            else:
                print(f"✗ Failed to create {model_name}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error in model creation test: {e}")
        return False

def test_data_loading():
    """Test data loading"""
    print("\nTesting data loading...")
    
    try:
        from comprehensive_evaluator import ComprehensiveModelEvaluator
        
        evaluator = ComprehensiveModelEvaluator()
        data = evaluator.load_data()
        
        if data is not None:
            print(f"✓ Data loaded successfully: {len(data)} records")
            print(f"  Date range: {data['date'].min()} to {data['date'].max()}")
            return True
        else:
            print("✗ Failed to load data")
            return False
            
    except Exception as e:
        print(f"✗ Error in data loading test: {e}")
        return False

def test_single_model_evaluation():
    """Test single model evaluation"""
    print("\nTesting single model evaluation...")
    
    try:
        from comprehensive_evaluator import ComprehensiveModelEvaluator
        
        evaluator = ComprehensiveModelEvaluator()
        data = evaluator.load_data()
        
        if data is None:
            print("✗ No data available for evaluation test")
            return False
        
        # Test with a simple technical model
        result = evaluator.evaluate_single_model('ma_cross_5_20', data, '1d')
        
        if 'error' not in result:
            print(f"✓ Model evaluation successful: Accuracy={result.get('accuracy_mean', 0):.4f}")
            return True
        else:
            print(f"✗ Model evaluation failed: {result['error']}")
            return False
            
    except Exception as e:
        print(f"✗ Error in model evaluation test: {e}")
        return False

def main():
    """Main test function"""
    print("="*60)
    print("BENCHMARK SYSTEM QUICK TEST")
    print("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("Model Creation Test", test_model_creation),
        ("Data Loading Test", test_data_loading),
        ("Model Evaluation Test", test_single_model_evaluation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")
    
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    print(f"{'='*60}")
    
    if passed == total:
        print("🎉 All tests passed! The benchmark system is ready to use.")
        print("\nTo run the full comparison:")
        print("python main_benchmark_comparison.py")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
