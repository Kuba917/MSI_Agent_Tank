#!/usr/bin/env python3
"""
Quick test script to verify CometML credentials and connectivity.
Run this to check if CometML is working before running the full training.
"""

import comet_ml

API_KEY = "L2PzW7c3YM3WqM5hNfCsloeLZ"
PROJECT_NAME = "msi-projekt"
WORKSPACE = "kluski777"

print("Testing CometML connection...")
print(f"Workspace: {WORKSPACE}")
print(f"Project: {PROJECT_NAME}")

try:
    experiment = comet_ml.start(
        api_key=API_KEY,
        project_name=PROJECT_NAME,
        workspace=WORKSPACE
    )
    
    experiment.set_name("Test_Connection")
    
    # Log some test data
    experiment.log_parameter("test_param", 42)
    experiment.log_metric("test_metric", 3.14, step=1)
    experiment.log_metric("test_metric", 2.71, step=2)
    
    print(f"\n✓ SUCCESS!")
    print(f"Experiment URL: {experiment.url}")
    print(f"Experiment Key: {experiment.get_key()}")
    
    # Flush and end
    experiment.flush()
    experiment.end()
    
    print("\n✓ Data uploaded successfully!")
    print(f"Check your experiment at: {experiment.url}")
    
except Exception as e:
    print(f"\n✗ FAILED!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
