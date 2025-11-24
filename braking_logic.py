import math
import io
import os
import shutil
import subprocess
import tempfile
from typing import Dict, Any
from jinja2 import Template

# Basic braking logic adapted from provided template and braking data
g = 9.81

def compute_braking(mass_kg: float, speed_kmh: float, mu: float = 0.3, reaction_time: float = 1.0, gradient: float = 0.0, gradient_type: str = 'percent', num_wheels: int = 4, max_braking_force: float = None) -> Dict[str, Any]:
    """
    Compute braking results for a single speed and return a dict with results.
    - mass_kg: vehicle mass in kg
    - speed_kmh: speed in km/h
    - mu: coefficient of friction (typical 0.25-0.4)
    - reaction_time: seconds
    - gradient: numeric gradient. Interpretation depends on gradient_type:
        - 'percent' => gradient in % (e.g. 1.5 => 1.5%)
        - '1in' => gradient given as '1 in N' value (N)
        - 'degree' => gradient in degrees
    - num_wheels: number of wheels (for per-wheel force)
    - max_braking_force: optional limit of braking force (N)
    """
    # conversions
    v_ms = speed_kmh * 1000.0 / 3600.0
    weight_n = mass_kg * g

    # reaction distance
    reaction_distance = v_ms * reaction_time

    # gradient angle (radians)
    if gradient_type == 'percent':
        # percent p => angle = arctan(p/100)
        angle = math.atan2(gradient, 100.0)
    elif gradient_type == '1in':
        # gradient given as '1 in N' -> value is N => rise/run = 1/N
        if gradient == 0:
            angle = 0.0
        else:
            angle = math.atan2(1.0, gradient)
    elif gradient_type == 'degree':
        angle = math.radians(gradient)
    else:
        angle = 0.0

    # gravitational force component along slope (positive downhill)
    fg = weight_n * math.sin(angle)

    # ideal braking deceleration using mu (d = v^2 / (2 * mu * g))
    # braking_accel magnitude (positive number)
    if mu > 0:
        a_brake = (v_ms * v_ms) / (2.0 * (v_ms * 0 + 1.0))  # placeholder to keep formula intent
    else:
        a_brake = 0.0

    # Instead compute braking deceleration from friction: a = mu * g
    deceleration = mu * g

    # braking distance (kinematics): v^2 = 2 * a * d  => d = v^2 / (2*a)
    if deceleration > 0:
        braking_distance = (v_ms ** 2) / (2.0 * deceleration)
    else:
        braking_distance = float('inf')

    # net effect of gravity: if moving downhill gravity reduces net deceleration
    # net_accel = deceleration - (fg / mass)
    net_accel = deceleration - (fg / mass_kg)
    if net_accel <= 0:
        total_stop_dist = float('inf')
    else:
        braking_distance_net = (v_ms ** 2) / (2.0 * net_accel)
        total_stop_dist = reaction_distance + braking_distance_net

    # braking force required (F = m * a). Use net deceleration for stopping
    braking_force = mass_kg * deceleration
    if max_braking_force is not None and max_braking_force > 0:
        # clamp braking force and recompute deceleration/distance
        if braking_force > max_braking_force:
            braking_force = max_braking_force
            deceleration = braking_force / mass_kg
            if deceleration > 0:
                braking_distance = (v_ms ** 2) / (2.0 * deceleration)
            else:
                braking_distance = float('inf')
            net_accel = deceleration - (fg / mass_kg)
            if net_accel > 0:
                braking_distance_net = (v_ms ** 2) / (2.0 * net_accel)
                total_stop_dist = reaction_distance + braking_distance_net
            else:
                total_stop_dist = float('inf')

    per_wheel_force = braking_force / max(1, num_wheels)

    return {
        'mass_kg': mass_kg,
        'speed_kmh': speed_kmh,
        'speed_ms': round(v_ms, 3),
        'reaction_distance_m': round(reaction_distance, 3),
        'braking_distance_m': round(braking_distance, 3),
        'deceleration_m_s2': round(deceleration, 4),
        'net_acceleration_m_s2': round(net_accel, 4) if isinstance(net_accel, float) else net_accel,
        'total_stopping_distance_m': (round(total_stop_dist, 3) if total_stop_dist != float('inf') else 'inf'),
        'braking_force_N': round(braking_force, 2),
        'per_wheel_force_N': round(per_wheel_force, 2),
        'gravity_component_N': round(fg, 2),
    }


def create_braking_pdf(inputs: Dict[str, Any], result: Dict[str, Any]) -> io.BytesIO:
    """Render `template.tex` with inputs+result, run pdflatex, and return PDF bytes in BytesIO.

    Requires `pdflatex` (TeX Live) available in PATH. Copies required images into the tempdir.
    """
    # locate template
    base_dir = os.path.dirname(__file__)
    template_path = os.path.join(base_dir, 'template.tex')
    if not os.path.exists(template_path):
        raise FileNotFoundError('template.tex not found')

    with open(template_path, 'r', encoding='utf-8') as f:
        tpl_text = f.read()

    # Prepare data for template
    # Map expected template variables to our values
    mass_kg = inputs.get('mass_kg')
    weight_n = round(mass_kg * g, 2)
    speed_kmh = inputs.get('speed_kmh')
    v_ms = round(speed_kmh * 1000.0 / 3600.0, 3)
    reference_braking_dist = result.get('braking_distance_m')
    decel = result.get('deceleration_m_s2')
    Reaction_distance = result.get('reaction_distance_m')
    totl_sto_distan = result.get('total_stopping_distance_m')
    fb = result.get('braking_force_N')
    gradient_input = inputs.get('gradient', 0)
    gradient_type = inputs.get('gradient_type', 'percent')
    # compute angle in degrees for template
    if gradient_type == 'percent':
        angle_deg = round(math.degrees(math.atan2(gradient_input, 100.0)), 3)
    elif gradient_type == '1in':
        angle_deg = round(math.degrees(math.atan2(1.0, gradient_input if gradient_input != 0 else 1)), 3)
    elif gradient_type == 'degree':
        angle_deg = gradient_input
    else:
        angle_deg = 0

    ctx = {
        'mass_kg': mass_kg,
        'weight_n': weight_n,
        'speed_kmh': speed_kmh,
        'v_ms': v_ms,
        'reference_braking_dist': reference_braking_dist,
        'decel': decel,
        'Reaction_distance': Reaction_distance,
        'totl_sto_distan': totl_sto_distan,
        'fb': fb,
        'gradient_input': gradient_input,
        'gradient_type': gradient_type,
        'angle_deg': angle_deg,
        # provide a small table with single row for the template loop
        'old_data_for_report': {speed_kmh: {
            'speed_ms': v_ms,
            'braking_distance': reference_braking_dist,
            'deceleration': decel,
            'reaction_distance': Reaction_distance,
            'total_stopping_distance': totl_sto_distan,
            'braking_force': fb
        }}
    }

    rendered = Template(tpl_text).render(**ctx)

    # create temp dir and write tex file + copy images
    tmp = tempfile.mkdtemp(prefix='brake_pdf_')
    try:
        tex_path = os.path.join(tmp, 'report.tex')
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(rendered)

        # copy images from base_dir (logo.JPG, logo-1.JPG, breaking distance table.png) if present
        for name in ['logo.JPG', 'logo-1.JPG', 'breaking distance table.png', 'breaking distance table.png', 'breaking distance table.png']:
            src = os.path.join(base_dir, name)
            if os.path.exists(src):
                try:
                    shutil.copy(src, tmp)
                except Exception:
                    pass

        # run pdflatex
        proc = subprocess.run(['pdflatex', '-interaction=nonstopmode', 'report.tex'], cwd=tmp, capture_output=True, timeout=30)
        if proc.returncode != 0:
            raise RuntimeError(f'pdflatex failed: {proc.stderr.decode("utf-8", errors="ignore")[:1000]}')

        pdf_path = os.path.join(tmp, 'report.pdf')
        if not os.path.exists(pdf_path):
            raise FileNotFoundError('PDF not generated')

        bio = io.BytesIO()
        with open(pdf_path, 'rb') as f:
            bio.write(f.read())
        bio.seek(0)
        return bio
    finally:
        try:
            shutil.rmtree(tmp)
        except Exception:
            pass
