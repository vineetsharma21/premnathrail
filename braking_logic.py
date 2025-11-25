"""
Braking Performance Calculator - Backend Logic
Based on DIN EN 15746-2:2021-05 standards for railway vehicles
"""

import math
from typing import Dict, Any, List


# =============================================================================
# CONSTANTS
# =============================================================================
g = 9.81  # Gravitational acceleration (m/s²)

# Standard braking distances for different speeds (DIN EN 15746-2:2021-05)
BRAKING_DATA = {
    8: 3, 10: 5, 16: 12, 20: 20, 24: 28,
    30: 45, 32: 50, 40: 75, 50: 135, 60: 180
}

# Maximum allowed total stopping distances for different speeds
MAX_STOPPING_DISTANCES = {
    8: 6, 10: 9, 16: 18, 20: 27, 24: 36,
    30: 55, 32: 60, 40: 90, 50: 155, 60: 230,
    70: 300, 80: 400, 90: 500, 100: 620
}


# =============================================================================
# GRADIENT CONVERSION FUNCTIONS
# =============================================================================
def convert_gradient_to_angle(gradient_value: float, gradient_type: str) -> float:
    """
    Convert gradient to angle in degrees.
    
    Args:
        gradient_value: The gradient value
        gradient_type: Type of gradient - 'degree', '1in', or 'percent'
    
    Returns:
        Angle in degrees
    """
    if gradient_type == "degree":
        return gradient_value
    elif gradient_type == "1in":
        if gradient_value == 0:
            return 0.0
        return round(math.degrees(math.atan(1 / gradient_value)), 4)
    elif gradient_type == "percent":
        return round(math.degrees(math.atan(gradient_value / 100)), 4)
    else:
        raise ValueError(f"Invalid gradient type: {gradient_type}")


# =============================================================================
# CORE CALCULATION FUNCTIONS
# =============================================================================
def calculate_braking_performance(
    mass_kg: float,
    speed_kmh: float,
    mu: float = 0.3,
    reaction_time: float = 1.0,
    gradient: float = 0.0,
    gradient_type: str = "percent",
    num_wheels: int = 4
) -> Dict[str, Any]:
    """
    Calculate braking performance for a vehicle.
    
    Args:
        mass_kg: Vehicle mass in kg
        speed_kmh: Speed in km/h
        mu: Coefficient of friction
        reaction_time: Reaction time in seconds
        gradient: Gradient value
        gradient_type: Type of gradient ('degree', '1in', or 'percent')
        num_wheels: Number of wheels
    
    Returns:
        Dictionary containing all calculation results
    """
    # Convert speed to m/s
    speed_ms = round(speed_kmh * (1000 / 3600), 2)
    
    # Convert gradient to angle
    angle_deg = convert_gradient_to_angle(gradient, gradient_type)
    angle_rad = math.radians(angle_deg)
    
    # Weight and gravitational force
    weight_n = round(mass_kg * g, 2)
    
    # Gravitational force component along the slope
    fg_n = round(weight_n * math.sin(angle_rad), 2)
    
    # Maximum braking force based on friction
    max_braking_force = round(mu * weight_n, 2)
    
    # Net braking force (considering gradient)
    if gradient > 0:  # Uphill
        net_force = round(max_braking_force + fg_n, 2)
    else:  # Downhill or flat
        net_force = round(max_braking_force - fg_n, 2)
    
    # Deceleration
    deceleration = round(net_force / mass_kg, 4) if mass_kg > 0 else 0.0
    
    # Reaction distance
    reaction_distance = round(speed_ms * reaction_time, 2)
    
    # Braking distance
    if deceleration > 0:
        braking_distance = round((speed_ms ** 2) / (2 * deceleration), 2)
    else:
        braking_distance = float('inf')
    
    # Total stopping distance
    if braking_distance != float('inf'):
        total_stopping_distance = round(reaction_distance + braking_distance, 2)
    else:
        total_stopping_distance = float('inf')
    
    # Braking force per wheel
    braking_force_per_wheel = round(max_braking_force / num_wheels, 2) if num_wheels > 0 else 0.0
    
    # Standard compliance check
    standard_compliance = check_standard_compliance(speed_kmh, total_stopping_distance)
    
    return {
        "speed_kmh": speed_kmh,
        "speed_ms": speed_ms,
        "mass_kg": mass_kg,
        "weight_n": weight_n,
        "mu": mu,
        "reaction_time": reaction_time,
        "gradient": gradient,
        "gradient_type": gradient_type,
        "angle_deg": angle_deg,
        "num_wheels": num_wheels,
        "gravitational_force_n": fg_n,
        "max_braking_force_n": max_braking_force,
        "net_force_n": net_force,
        "deceleration_m_s2": deceleration,
        "reaction_distance_m": reaction_distance,
        "braking_distance_m": braking_distance,
        "total_stopping_distance_m": total_stopping_distance,
        "braking_force_per_wheel_n": braking_force_per_wheel,
        "standard_compliance": standard_compliance
    }


def calculate_multi_speed_analysis(
    mass_kg: float,
    max_speed_kmh: float,
    speed_increment: float,
    mu: float = 0.3,
    reaction_time: float = 1.0,
    gradient: float = 0.0,
    gradient_type: str = "percent",
    num_wheels: int = 4
) -> List[Dict[str, Any]]:
    """
    Calculate braking performance for multiple speeds.
    
    Args:
        mass_kg: Vehicle mass in kg
        max_speed_kmh: Maximum speed in km/h
        speed_increment: Speed increment for analysis
        mu: Coefficient of friction
        reaction_time: Reaction time in seconds
        gradient: Gradient value
        gradient_type: Type of gradient
        num_wheels: Number of wheels
    
    Returns:
        List of calculation results for each speed
    """
    results = []
    current_speed = speed_increment
    
    while current_speed <= max_speed_kmh:
        result = calculate_braking_performance(
            mass_kg=mass_kg,
            speed_kmh=current_speed,
            mu=mu,
            reaction_time=reaction_time,
            gradient=gradient,
            gradient_type=gradient_type,
            num_wheels=num_wheels
        )
        results.append(result)
        current_speed += speed_increment
    
    return results


def calculate_multi_gradient_analysis(
    mass_kg: float,
    speed_kmh: float,
    max_gradient: float,
    gradient_steps: int,
    gradient_type: str = "percent",
    mu: float = 0.3,
    reaction_time: float = 1.0,
    num_wheels: int = 4
) -> List[Dict[str, Any]]:
    """
    Calculate braking performance for multiple gradients.
    
    Args:
        mass_kg: Vehicle mass in kg
        speed_kmh: Speed in km/h
        max_gradient: Maximum gradient value
        gradient_steps: Number of gradient steps
        gradient_type: Type of gradient
        mu: Coefficient of friction
        reaction_time: Reaction time in seconds
        num_wheels: Number of wheels
    
    Returns:
        List of calculation results for each gradient
    """
    results = []
    gradient_increment = max_gradient / gradient_steps if gradient_steps > 0 else 0
    
    # Include flat surface (0 gradient)
    result = calculate_braking_performance(
        mass_kg=mass_kg,
        speed_kmh=speed_kmh,
        mu=mu,
        reaction_time=reaction_time,
        gradient=0.0,
        gradient_type=gradient_type,
        num_wheels=num_wheels
    )
    results.append(result)
    
    # Calculate for incremental gradients
    for i in range(1, gradient_steps + 1):
        current_gradient = gradient_increment * i
        result = calculate_braking_performance(
            mass_kg=mass_kg,
            speed_kmh=speed_kmh,
            mu=mu,
            reaction_time=reaction_time,
            gradient=current_gradient,
            gradient_type=gradient_type,
            num_wheels=num_wheels
        )
        results.append(result)
    
    return results


def check_standard_compliance(speed_kmh: float, calculated_distance: float) -> str:
    """
    Check if the calculated stopping distance complies with EN standard.
    
    Args:
        speed_kmh: Speed in km/h
        calculated_distance: Calculated total stopping distance in meters
    
    Returns:
        Compliance status string
    """
    if calculated_distance == float('inf'):
        return "✗ Physical Limit Exceeded"
    
    # Find the relevant standard limit for the given speed
    allowed_distance = None
    for speed_limit in sorted(MAX_STOPPING_DISTANCES.keys(), reverse=True):
        if speed_kmh >= speed_limit:
            allowed_distance = MAX_STOPPING_DISTANCES[speed_limit]
            break
    
    if allowed_distance is None:
        return "Standard Not Found"
    
    if calculated_distance <= allowed_distance:
        return "✓ Standard Followed"
    else:
        return "✗ Standard Exceeded"


def calculate_gbr(braking_force_n: float, mass_kg: float) -> float:
    """
    Calculate Global Braking Ratio (GBR).
    GBR = Braking Force / (Mass × g)
    
    Args:
        braking_force_n: Braking force in Newtons
        mass_kg: Mass in kg
    
    Returns:
        GBR value
    """
    if mass_kg <= 0:
        return 0.0
    return round(braking_force_n / (mass_kg * g), 4)


# =============================================================================
# REPORT GENERATION HELPERS
# =============================================================================
def format_result_for_display(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format calculation result for display.
    
    Args:
        result: Raw calculation result
    
    Returns:
        Formatted result dictionary
    """
    return {
        "Speed (km/h)": result["speed_kmh"],
        "Speed (m/s)": result["speed_ms"],
        "Reaction Distance (m)": result["reaction_distance_m"],
        "Braking Distance (m)": result["braking_distance_m"],
        "Total Stopping Distance (m)": result["total_stopping_distance_m"],
        "Deceleration (m/s²)": result["deceleration_m_s2"],
        "Braking Force (N)": result["max_braking_force_n"],
        "Net Force (N)": result["net_force_n"],
        "Standard Compliance": result["standard_compliance"]
    }


def get_standard_table_data() -> List[Dict[str, Any]]:
    """
    Get the standard braking distance table data.
    
    Returns:
        List of standard data entries
    """
    table_data = []
    for speed, braking_dist in sorted(BRAKING_DATA.items()):
        max_stop_dist = MAX_STOPPING_DISTANCES.get(speed, "N/A")
        table_data.append({
            "speed_kmh": speed,
            "braking_distance_m": braking_dist,
            "max_stopping_distance_m": max_stop_dist
        })
    return table_data
