import numpy as np
import cara_py
import os

def main():
    print("--- CARA SDK CDM Parsing Example ---")
    
    cdm_path = "upstream/DataFiles/SampleCDMs/AlfanoTestCase01.cdm"
    
    if not os.path.exists(cdm_path):
        print(f"Error: Could not find CDM file at {cdm_path}")
        return

    # 1. Parse CDM
    print(f"Parsing CDM: {cdm_path}")
    cdm = cara_py.parse_cdm(cdm_path)
    
    print(f"CDM Version: {cdm.header.version}")
    print(f"TCA:         {cdm.header.tca}")
    print(f"Miss Dist:   {cdm.header.miss_distance} m")
    print(f"HBR:         {cdm.header.hbr} m")
    print(f"Object 1:    {cdm.object1.object_name}")
    print(f"Object 2:    {cdm.object2.object_name}")

    # 2. Compute PC directly from CDM path
    pc = cara_py.compute_pc_from_cdm(cdm_path)
    print(f"\nCalculated Pc from CDM (Foster): {pc:.4e}")

if __name__ == "__main__":
    main()
