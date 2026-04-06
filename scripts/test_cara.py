import numpy as np
import cara_py

def main():
    """
    Demonstrate basic usage of the CARA Rust SDK Python bindings.
    """
    print("--- CARA SDK Python Example ---")

    # 1. Define states for two objects (Primary and Secondary)
    # Positions in km, velocities in km/s
    r1 = np.array([7000.0, 0.0, 0.0])
    v1 = np.array([0.0, 7.5, 0.0])
    cov1 = np.diag([0.01, 0.01, 0.01]) # 3x3 position covariance in km^2
    
    r2 = np.array([7000.0, 0.001, 0.0])
    v2 = np.array([0.0, 0.0, 7.5])
    cov2 = np.diag([0.01, 0.01, 0.01])
    
    # 2. Hard Body Radius (HBR) in km
    hbr = 0.01 # 10 meters
    
    # 3. Calculate 2D Pc using Foster's Method
    pc_foster = cara_py.compute_2d_foster(r1, v1, cov1, r2, v2, cov2, hbr)
    print(f"2D Pc (Foster): {pc_foster:.4e}")

    # 4. Calculate 2D Pc using high-precision Circle Method (Gauss-Chebyshev)
    pc_circle = cara_py.compute_pc_circle(r1, v1, cov1, r2, v2, cov2, hbr)
    print(f"2D Pc (Circle): {pc_circle:.4e}")

    # 4b. Calculate 2D Pc using Elrod Method
    pc_elrod = cara_py.compute_pc_elrod(r1, v1, cov1, r2, v2, cov2, hbr)
    print(f"2D Pc (Elrod):  {pc_elrod:.4e}")

    # 4c. Calculate Pc using SDMC (Monte Carlo)
    # Using a 6x6 covariance by padding the 3x3 position covariance
    cov1_6x6 = np.zeros((6, 6))
    cov1_6x6[:3, :3] = cov1
    cov2_6x6 = np.zeros((6, 6))
    cov2_6x6[:3, :3] = cov2
    pc_sdmc = cara_py.compute_pc_sdmc(r1, v1, cov1_6x6, r2, v2, cov2_6x6, hbr, 100000, 42)
    print(f"Pc (SDMC):      {pc_sdmc:.4e}")

    # 4d. Calculate Frisbee Max Pc
    max_pc_frisbee = cara_py.compute_frisbee_max_pc(r1, v1, cov1, r2, v2, cov2, hbr)
    print(f"Max Pc (Frisbee): {max_pc_frisbee:.4e}")

    # 4e. Calculate Pc Dilution
    dil_out = cara_py.compute_pc_dilution(r1, v1, cov1, r2, v2, cov2, hbr, "both")
    print(f"Pc Dilution:")
    print(f"  Nominal Pc: {dil_out.pc_one:.4e}")
    print(f"  Max Pc:     {dil_out.pc_max:.4e}")
    print(f"  Sf at Max:  {dil_out.sf_max:.4f}")
    print(f"  Diluted:    {dil_out.diluted}")

    # 5. Rotate ECI covariance to RIC frame
    ric_cov = cara_py.rotate_eci_to_ric(cov1, r1, v1)
    print("\nRIC Covariance Matrix:")
    print(ric_cov)

    # 6. Convert Cartesian state to Keplerian elements
    # State: [x, y, z, vx, vy, vz]
    cart = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    kep = cara_py.cart_to_keplerian(cart)
    print(f"\nKeplerian elements for circular LEO orbit:")
    print(f"  Semi-major axis (a): {kep.a:.2f} km")
    print(f"  Eccentricity (e):    {kep.e:.4f}")
    print(f"  Inclination (i):     {kep.i:.2f} deg")

    # 7. Julian Date conversion
    timestring = "2000-01-01 12:00:00"
    jd = cara_py.get_timestring_to_jd(timestring)
    print(f"\nJulian Date for {timestring}: {jd}")

if __name__ == "__main__":
    main()
