#!/usr/bin/env python3
"""
Main Execution Script - Complete Ensemble Analysis Pipeline
==========================================================

This script orchestrates the complete ensemble analysis pipeline:
1. Automatically generate missing score files (AAPL, VGT, Sentiment) if needed
2. Perform ensemble weight optimization with grid search
3. Generate comprehensive visualizations
4. Print final results and best weight combinations

The script will automatically detect missing score files and generate them,
making it a true one-click solution.

Date: 2025
"""

import os
import sys
import time
from datetime import datetime

def get_project_root():
    """Get the project root directory"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def print_header():
    """Print analysis header"""
    print("="*80)
    print("COMPREHENSIVE ENSEMBLE ANALYSIS PIPELINE")
    print("="*80)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

def print_section_header(section_name):
    """Print section header"""
    print(f"\n{'='*20} {section_name} {'='*20}")

def run_ensemble_analysis():
    """Run ensemble analysis with weight optimization"""
    print_section_header("ENSEMBLE WEIGHT OPTIMIZATION")
    
    try:
        print("Running ensemble weight optimization with grid search...")
        from .score_ensemble import ScoreEnsemble
        
        ensemble = ScoreEnsemble()
        results = ensemble.run_full_analysis()
        
        if results is not None:
            print("✓ Ensemble weight optimization completed successfully!")
            
            # Print summary
            print("\nENSEMBLE WEIGHT OPTIMIZATION SUMMARY:")
            print("-" * 50)
            
            # Find best performing horizon
            best_horizon = None
            best_accuracy = 0
            
            for horizon in ['1d', '3d', '5d']:
                if horizon in ensemble.best_models:
                    model_info = ensemble.best_models[horizon]
                    accuracy = model_info['test_metrics']['accuracy']
                    
                    print(f"{horizon.upper()} Horizon:")
                    print(f"  Best Weights: AAPL={model_info['weights']['aapl_weight']:.3f}, "
                          f"VGT={model_info['weights']['vgt_weight']:.3f}, "
                          f"Sentiment={model_info['weights']['sentiment_weight']:.3f}")
                    print(f"  Test Accuracy: {accuracy:.4f}")
                    print(f"  Test F1-Score: {model_info['test_metrics']['f1_score']:.4f}")
                    print()
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_horizon = horizon
            
            print(f"🏆 BEST PERFORMING HORIZON: {best_horizon.upper()}")
            print(f"   Accuracy: {best_accuracy:.4f}")
            
            return ensemble
        else:
            print("✗ Ensemble analysis failed")
            return None
            
    except Exception as e:
        print(f"✗ Error in ensemble analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_visualization():
    """Run comprehensive visualization"""
    print_section_header("VISUALIZATION")
    
    try:
        print("Generating comprehensive visualizations...")
        from .visualization import ResultsVisualizer
        
        visualizer = ResultsVisualizer()
        visualizer.create_comprehensive_report()
        
        print("✓ Visualization completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error in visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_final_summary(ensemble):
    """Print final analysis summary"""
    print_section_header("FINAL SUMMARY")
    
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("-" * 40)
    
    # File outputs
    print("\nGenerated Files:")
    from config import (
        ENSEMBLE_OUTPUT_DIR,
        AAPL_SCORES, VGT_SCORES, SENTIMENT_SCORES,
        COMBINED_SCORES, ENSEMBLE_RESULTS, ENSEMBLE_SUMMARY
    )
    file_map = [
        ("aapl_scores.csv", str(AAPL_SCORES)),
        ("vgt_scores.csv", str(VGT_SCORES)),
        ("sentiment_scores.csv", str(SENTIMENT_SCORES)),
        ("combined_scores.csv", str(COMBINED_SCORES)),
        ("ensemble_results.pkl", str(ENSEMBLE_RESULTS)),
        ("ensemble_summary.txt", str(ENSEMBLE_SUMMARY)),
    ]
    
    for label, file_path in file_map:
        if os.path.exists(file_path):
            print(f"✓ {label}")
        else:
            print(f"✗ {label} (not found)")
    
    # Visualization files
    from config import ENSEMBLE_OUTPUT_DIR
    results_dir = str(ENSEMBLE_OUTPUT_DIR)
    if os.path.exists(results_dir):
        viz_files = [
            "model_comparison.png",
            "ensemble_performance.png",
            "time_series_analysis.png",
            "grid_search_analysis.png",
            "final_results_summary.png",
            "summary_statistics.txt"
        ]
        
        print("\nVisualization Files:")
        for file in viz_files:
            file_path = os.path.join(results_dir, file)
            if os.path.exists(file_path):
                print(f"✓ {file}")
            else:
                print(f"✗ {file} (not found)")
    
    # Best weight combination summary
    if ensemble and hasattr(ensemble, 'best_models'):
        print("\nBEST WEIGHT COMBINATIONS BY HORIZON:")
        print("-" * 50)
        print(f"{'Horizon':<8} {'AAPL':<8} {'VGT':<8} {'Sentiment':<10} {'Accuracy':<10}")
        print("-" * 50)
        
        best_horizon = None
        best_accuracy = 0
        
        for horizon in ['1d', '3d', '5d']:
            if horizon in ensemble.best_models:
                model_info = ensemble.best_models[horizon]
                weights = model_info['weights']
                accuracy = model_info['test_metrics']['accuracy']
                
                print(f"{horizon:<8} {weights['aapl_weight']:<8.3f} {weights['vgt_weight']:<8.3f} "
                      f"{weights['sentiment_weight']:<10.3f} {accuracy:<10.4f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_horizon = horizon
        
        print("-" * 50)
        print(f"\n🏆 BEST PERFORMING HORIZON: {best_horizon.upper()}")
        print(f"   Accuracy: {best_accuracy:.4f}")
    
    print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

def main():
    """Main execution function - Complete pipeline"""
    start_time = time.time()
    
    # Print header
    print_header()
    
    # Step 1: Run ensemble analysis (automatically handles missing files)
    ensemble = run_ensemble_analysis()
    if ensemble is None:
        print("\n✗ Pipeline failed at ensemble analysis step")
        return False
    
    # Step 2: Run visualization
    if not run_visualization():
        print("\n✗ Pipeline failed at visualization step")
        return False
    
    # Step 3: Print final summary
    print_final_summary(ensemble)
    
    # Calculate total time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    
    print("\n🎉 COMPLETE ENSEMBLE ANALYSIS PIPELINE FINISHED SUCCESSFULLY! 🎉")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
