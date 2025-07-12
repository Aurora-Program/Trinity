#!/usr/bin/env python3
"""Debug validation to see which test is failing"""

from Trinity_Fixed_Complete import Transcender

def debug_validation():
    transcender = Transcender()
    Ms, Ss, MetaM = transcender.procesar([1,0,1], [0,1,0], [1,1,0])
    
    run_data = transcender.last_run_data
    
    # All validations
    validations = {
        "intermediate_exists": "intermediate" in run_data,
        "no_logic_key": "logic" not in run_data,
        "s_values_exist": False,
        "s_values_valid": False,
        "ms_structure": isinstance(Ms, list) and len(Ms) == 3,
        "ss_structure": isinstance(Ss, list) and len(Ss) == 3,
        "metam_structure": isinstance(MetaM, list) and len(MetaM) == 4
    }
    
    if validations["intermediate_exists"]:
        intermediate = run_data["intermediate"]
        validations["s_values_exist"] = all(key in intermediate for key in ["S1", "S2", "S3"])
        validations["s_values_valid"] = all(isinstance(intermediate.get(key), list) and len(intermediate.get(key)) == 3 
                                          for key in ["S1", "S2", "S3"] if key in intermediate)
    
    print("Detailed validation results:")
    for test_name, result in validations.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:.<30} {status}")
    
    print(f"\nData structure:")
    print(f"  run_data keys: {list(run_data.keys())}")
    print(f"  intermediate: {run_data.get('intermediate', 'NOT FOUND')}")
    print(f"  logic exists: {'logic' in run_data}")

if __name__ == "__main__":
    debug_validation()
