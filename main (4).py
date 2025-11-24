import math
import io
import os
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

# --- Database & Security Imports ---
from sqlalchemy.orm import Session
from passlib.context import CryptContext
import models
import schemas
from database import engine, get_db

# Optional imports
try:
    import docx
except ImportError:
    docx = None

# --- Import Logic ---
from Hydraulic_Motor_Calculation import Calculator as HydraulicCalculator
from QmaxCalculator_Logic import QmaxCalculatorLogic, SIGMA_B_OPTIONS
from LoadDistribution_Logic import perform_calculations as perform_load_distro_calc
from LoadDistribution_Logic import format_detailed_steps as format_load_distro_steps
from LoadDistribution_Logic import create_report_docx as create_load_distro_docx
from Tractive_Effort_Logic import perform_te_calculations, format_te_report_text, create_te_report_docx
from Vehicle_Performance_Logic import VehiclePerformanceCalculator

# --- DATABASE SETUP ---
models.Base.metadata.create_all(bind=engine)

# --- SECURITY SETUP ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# ---------- Pydantic Models ----------
class HydraulicRawInput(BaseModel):
    calc_mode: str
    weight: str
    axles: str
    speed: str
    max_vehicle_rpm: str
    pto_gear_ratio: str
    engine_gear_ratio: str
    axle_gear_box_ratio: str
    slope_percent: str
    curve_degree: str
    wheel_diameter: str
    num_motors: str
    per_axle_motor: str
    pressure: str
    mech_eff_motor: str
    vol_eff_motor: str
    motor_disp_in: str
    max_motor_rpm: str
    vol_eff_pump: str
    pump_disp_in: str
    max_pump_rpm: str

class QmaxRawInput(BaseModel):
    d: str
    sigma_b_selection: str
    sigma_b_custom: str
    v_head: str

class LoadDistroRawInput(BaseModel):
    config_type: str
    total_load: str
    front_percent: str
    q1_percent: str
    q3_percent: str

class TractiveEffortRawInput(BaseModel):
    load: str
    loco_weight: str
    gradient: str
    curvature: str
    speed: str
    mode: str
    grad_type: str
    curvature_unit: str

class VehiclePerformanceRawInput(BaseModel):
    max_curve: str
    max_slope: str
    loco_gvw: str
    max_speed: str
    num_axles: str
    rear_axle_ratio: str
    gear_ratios: str
    shunting_load: str
    peak_power: str
    friction_mu: str
    wheel_dia: str
    min_rpm: str
    max_rpm: str
    torque_curve: Dict[str, float] = Field(default_factory=dict)

# ---------- Validation Functions ----------
def _validate_input(value_str: str, type_func, name: str, is_optional=False, default=0.0, is_disabled=False):
    if is_disabled: return default
    if not value_str:
        if is_optional: return default
        else: raise ValueError(f"'{name}' cannot be empty.")
    try:
        return type_func(value_str)
    except:
        raise ValueError(f"Invalid value for '{name}'")

# (Shortened validation wrappers for brevity - logic remains same as before)
def process_and_validate_hydraulic_inputs(raw: HydraulicRawInput):
    # ... (Your existing validation logic) ...
    # NOTE: Main logic wahi hai jo pichli baar tha, bas space bachane ke liye yahan short mein likh raha hu
    # Aap purana validation logic as-is rakh sakte hain
    # Lekin agar aap chahein to main poora function dobara likh deta hu:
    inputs = {}
    inputs_raw = raw.dict()
    mode = raw.calc_mode
    inputs['calc_mode'] = mode
    inputs['weight'] = _validate_input(raw.weight, float, "Vehicle Weight")
    inputs['axles'] = _validate_input(raw.axles, int, "Number of axles")
    inputs['num_motors'] = _validate_input(raw.num_motors, int, "Hydraulic Motor")
    inputs['per_axle_motor'] = _validate_input(raw.per_axle_motor, int, "Motor / axle")
    inputs['slope_percent'] = _validate_input(raw.slope_percent, float, "Slope", True)
    inputs['curve_degree'] = _validate_input(raw.curve_degree, float, "Curve", True)
    inputs['wheel_diameter'] = _validate_input(raw.wheel_diameter, float, "Wheel Dia")
    inputs['axle_gear_box_ratio'] = _validate_input(raw.axle_gear_box_ratio, float, "Axle Gear Ratio")
    
    engine_gear_raw = raw.engine_gear_ratio.strip()
    parts = [p.strip() for p in engine_gear_raw.split(',') if p.strip()]
    engine_gear_list = [_validate_input(p, float, "Engine Gear") for p in parts]
    inputs['engine_gear_ratio_list'] = engine_gear_list
    inputs['engine_gear_ratio'] = engine_gear_list[0] if engine_gear_list else 1.0
    
    inputs['max_vehicle_rpm'] = _validate_input(raw.max_vehicle_rpm, float, "Max RPM")
    inputs['pto_gear_ratio'] = _validate_input(raw.pto_gear_ratio, float, "PTO Ratio")
    inputs['vol_eff_motor'] = _validate_input(raw.vol_eff_motor, float, "Motor Vol Eff")
    inputs['vol_eff_pump'] = _validate_input(raw.vol_eff_pump, float, "Pump Vol Eff")
    
    is_cc = (mode == 'calc_cc')
    inputs['speed'] = _validate_input(raw.speed, float, "Speed", True, 0.0, not is_cc)
    inputs['pressure'] = _validate_input(raw.pressure, float, "Pressure", True, 0.0, not is_cc)
    inputs['mech_eff_motor'] = _validate_input(raw.mech_eff_motor, float, "Mech Eff", True, 0.0, not is_cc)
    inputs['motor_disp_in'] = _validate_input(raw.motor_disp_in, float, "Motor Disp", True, 0.0, is_cc)
    inputs['max_motor_rpm'] = _validate_input(raw.max_motor_rpm, float, "Max Motor RPM", True, 3000.0, is_cc)
    inputs['pump_disp_in'] = _validate_input(raw.pump_disp_in, float, "Pump Disp", True, 0.0, is_cc)
    inputs['max_pump_rpm'] = _validate_input(raw.max_pump_rpm, float, "Max Pump RPM", True, 3000.0, is_cc)
    
    return inputs, inputs_raw

def process_and_validate_qmax_inputs(raw: QmaxRawInput):
    inputs = {}
    inputs_raw = raw.dict()
    inputs['d'] = _validate_input(raw.d, float, "Worn rail diameter")
    inputs['v_head'] = _validate_input(raw.v_head, float, "Safety Factor")
    if raw.sigma_b_selection == "Custom":
        inputs['sigma_b'] = _validate_input(raw.sigma_b_custom, float, "Custom Sigma B")
    else:
        inputs['sigma_b'] = SIGMA_B_OPTIONS.get(raw.sigma_b_selection, 0)
    return inputs, inputs_raw

def process_and_validate_load_distro_inputs(raw: LoadDistroRawInput):
    inputs = {}
    inputs_raw = raw.dict()
    inputs['config_type'] = raw.config_type
    inputs['total_load'] = _validate_input(raw.total_load, float, "Total Load")
    inputs['front_percent'] = _validate_input(raw.front_percent, float, "Front %")
    inputs['q1_percent'] = _validate_input(raw.q1_percent, float, "Q1 %")
    inputs['q3_percent'] = _validate_input(raw.q3_percent, float, "Q3 %")
    return inputs, inputs_raw

def process_and_validate_te_inputs(raw: TractiveEffortRawInput):
    inputs = {}
    inputs_raw = raw.dict()
    inputs['load'] = _validate_input(raw.load, float, "Load")
    inputs['loco_weight'] = _validate_input(raw.loco_weight, float, "Loco Weight")
    inputs['gradient'] = _validate_input(raw.gradient, float, "Gradient")
    inputs['curvature'] = _validate_input(raw.curvature, float, "Curvature")
    inputs['speed'] = _validate_input(raw.speed, float, "Speed")
    inputs['mode'] = raw.mode
    inputs['grad_type'] = raw.grad_type
    inputs['curvature_unit'] = raw.curvature_unit
    return inputs, inputs_raw

def process_and_validate_vehicle_performance_inputs(raw: VehiclePerformanceRawInput):
    inputs = {}
    inputs_raw = raw.dict()
    inputs['max_curve'] = _validate_input(raw.max_curve, float, "Max Curve", True)
    inputs['curve_unit'] = 'degree'
    inputs['max_slope'] = _validate_input(raw.max_slope, float, "Max Slope", True)
    inputs['slope_unit'] = '%'
    inputs['loco_gvw_kg'] = _validate_input(raw.loco_gvw, float, "Loco GVW")
    inputs['max_speed_kmh'] = _validate_input(raw.max_speed, float, "Max Speed", True)
    inputs['num_axles'] = _validate_input(raw.num_axles, int, "Axles")
    inputs['rear_axle_ratio'] = _validate_input(raw.rear_axle_ratio, float, "Rear Ratio")
    
    ratios = (raw.gear_ratios or "").split(',')
    inputs['gear_ratios'] = [_validate_input(r.strip(), float, "Gear Ratio") for r in ratios if r.strip()]
    if not inputs['gear_ratios']: inputs['gear_ratios'] = [1.0]
    
    inputs['shunting_load_t'] = _validate_input(raw.shunting_load, float, "Shunting Load", True)
    inputs['peak_power_kw'] = _validate_input(raw.peak_power, float, "Peak Power", True)
    inputs['friction_mu'] = _validate_input(raw.friction_mu, float, "Friction")
    inputs['wheel_dia_m'] = _validate_input(raw.wheel_dia, float, "Wheel Dia")
    inputs['min_rpm'] = _validate_input(raw.min_rpm, int, "Min RPM", True)
    inputs['max_rpm'] = _validate_input(raw.max_rpm, int, "Max RPM")
    
    tc = {}
    if raw.torque_curve:
        for k, v in raw.torque_curve.items():
            try: tc[int(k)] = float(v)
            except: pass
    inputs['torque_curve'] = tc
    return inputs, inputs_raw


# =======================================================
# ⚠️ CRITICAL: APP INITIALIZATION MUST BE HERE (BEFORE ROUTES)
# =======================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logic Instances
hydraulic_calculator = HydraulicCalculator()
qmax_calculator = QmaxCalculatorLogic()


# ==========================================
#      STATIC PAGE ROUTES (HTML Serving)
# ==========================================

@app.get("/")
async def serve_home():
    return FileResponse('index.html')

@app.get("/login")
async def serve_login_page():
    if os.path.exists("login.html"):
        return FileResponse('login.html')
    return {"error": "login.html file not found on server"}

@app.get("/signup")
async def serve_signup_page():
    if os.path.exists("signup.html"):
        return FileResponse('signup.html')
    return {"error": "signup.html file not found on server"}

@app.get("/profile")
async def serve_profile_page():
    if os.path.exists("profile.html"):
        return FileResponse('profile.html')
    return {"error": "profile.html file not found on server"}

@app.get("/calculator")
async def serve_hydraulic_calculator():
    return FileResponse('calculator.html')

@app.get("/qmax")
async def serve_qmax_page():
    return FileResponse('qmax_calculator.html')

@app.get("/load_distribution")
async def serve_load_distribution_page():
    return FileResponse('load_distribution_calculator.html')

@app.get("/tractive_effort")
async def serve_tractive_effort_page():
    return FileResponse('tractive_effort_calculator.html')

@app.get("/vehicle_performance")
async def serve_vehicle_performance_page():
    return FileResponse('vehicle_performance_calculator.html')

@app.get("/Diagram.png")
async def serve_diagram():
    if os.path.exists("Diagram.png"): return FileResponse('Diagram.png')
    raise HTTPException(status_code=404, detail="Not found")

@app.get("/logo.png")
async def serve_logo():
    if os.path.exists("logo.png"): return FileResponse('logo.png')
    raise HTTPException(status_code=404, detail="Not found")


# ==========================================
#      AUTH & API ROUTES
# ==========================================

@app.post("/signup")
def signup(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = get_password_hash(user.password)
    new_user = models.User(email=user.email, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "Account created successfully", "user_id": new_user.id}

@app.post("/login")
def login(user: schemas.UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid email or password")
    return {
        "message": "Login successful",
        "user_id": db_user.id,
        "email": db_user.email,
        "is_license_active": db_user.is_license_active
    }

@app.post("/activate_license")
def activate_license(activation: schemas.LicenseActivate, user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user: raise HTTPException(status_code=404, detail="User not found")
    
    license_entry = db.query(models.LicenseKey).filter(models.LicenseKey.key == activation.license_key).first()
    if not license_entry: raise HTTPException(status_code=400, detail="Invalid License Key")
    if license_entry.is_used: raise HTTPException(status_code=400, detail="Key already used")
        
    license_entry.is_used = True
    user.is_license_active = True
    db.commit()
    return {"message": "License Activated!"}

@app.post("/generate_key")
def generate_license_key(key_code: str, db: Session = Depends(get_db)):
    existing = db.query(models.LicenseKey).filter(models.LicenseKey.key == key_code).first()
    if existing: raise HTTPException(status_code=400, detail="Key exists")
    new_key = models.LicenseKey(key=key_code)
    db.add(new_key)
    db.commit()
    return {"message": f"Key '{key_code}' created"}


# ==========================================
#      CALCULATION ROUTES
# ==========================================

@app.post("/calculate")
async def handle_calculation(raw_input: HydraulicRawInput):
    try:
        inputs, inputs_raw = process_and_validate_hydraulic_inputs(raw_input)
        hydraulic_calculator.inputs_raw = inputs_raw
        if inputs['calc_mode'] == "calc_cc":
            res = hydraulic_calculator.perform_displacement_calculation(inputs)
            rep = hydraulic_calculator._generate_mode1_report(inputs, res)
        else:
            res = hydraulic_calculator.perform_speed_calculation(inputs)
            rep = hydraulic_calculator._generate_mode2_report(inputs, res)
        return {"report": rep, "results": res}
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/download_report")
async def download_report(raw_input: HydraulicRawInput):
    if docx is None: raise HTTPException(500, "python-docx not installed")
    try:
        inputs, inputs_raw = process_and_validate_hydraulic_inputs(raw_input)
        hydraulic_calculator.inputs_raw = inputs_raw
        doc = docx.Document()
        if inputs['calc_mode'] == "calc_cc":
            res = hydraulic_calculator.perform_displacement_calculation(inputs)
            hydraulic_calculator._create_mode1_docx(doc, inputs, res)
            fname = "Hydraulic_Report.docx"
        else:
            res = hydraulic_calculator.perform_speed_calculation(inputs)
            hydraulic_calculator._create_mode2_docx(doc, inputs, res)
            fname = "Speed_Report.docx"
        stream = io.BytesIO()
        doc.save(stream)
        stream.seek(0)
        return StreamingResponse(stream, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", headers={"Content-Disposition": f"attachment; filename={fname}"})
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/calculate_qmax")
async def handle_qmax(raw: QmaxRawInput):
    try:
        inp, raw_inp = process_and_validate_qmax_inputs(raw)
        qmax_calculator.inputs_raw = raw_inp
        res = qmax_calculator.perform_calculations(inp['d'], inp['sigma_b'], inp['v_head'])
        rep = qmax_calculator.format_detailed_steps(res)
        return {"report": rep, "results": res}
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/download_qmax_report")
async def dl_qmax(raw: QmaxRawInput):
    try:
        inp, raw_inp = process_and_validate_qmax_inputs(raw)
        qmax_calculator.inputs_raw = raw_inp
        res = qmax_calculator.perform_calculations(inp['d'], inp['sigma_b'], inp['v_head'])
        stream = qmax_calculator.create_report_docx(res)
        return StreamingResponse(stream, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", headers={"Content-Disposition": "attachment; filename=Qmax_Report.docx"})
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/calculate_load_distribution")
async def handle_load(raw: LoadDistroRawInput):
    try:
        inp, raw_inp = process_and_validate_load_distro_inputs(raw)
        res = perform_load_distro_calc(inp['config_type'], inp['total_load'], inp['front_percent'], inp['q1_percent'], inp['q3_percent'])
        rep = format_load_distro_steps(inp, res)
        return {"report": rep, "results": res}
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/download_load_distribution_report")
async def dl_load(raw: LoadDistroRawInput):
    try:
        inp, raw_inp = process_and_validate_load_distro_inputs(raw)
        res = perform_load_distro_calc(inp['config_type'], inp['total_load'], inp['front_percent'], inp['q1_percent'], inp['q3_percent'])
        stream = create_load_distro_docx(inp, res)
        return StreamingResponse(stream, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", headers={"Content-Disposition": "attachment; filename=Load_Report.docx"})
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/calculate_tractive_effort")
async def handle_te(raw: TractiveEffortRawInput):
    try:
        inp, raw_inp = process_and_validate_te_inputs(raw)
        res = perform_te_calculations(inp)
        rep = format_te_report_text(inp, res)
        return {"report": rep, "results": res}
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/download_tractive_effort_report")
async def dl_te(raw: TractiveEffortRawInput):
    try:
        inp, raw_inp = process_and_validate_te_inputs(raw)
        res = perform_te_calculations(inp)
        stream = create_te_report_docx(inp, res)
        return StreamingResponse(stream, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", headers={"Content-Disposition": "attachment; filename=TE_Report.docx"})
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/calculate_performance")
async def handle_perf(raw: VehiclePerformanceRawInput):
    try:
        inp, raw_inp = process_and_validate_vehicle_performance_inputs(raw)
        calc = VehiclePerformanceCalculator(inp)
        res = calc.run_tractive_calculation()
        plot = calc.calculate_plot_data()
        table = calc.calculate_speed_for_shunting_load()
        return {"traction_snapshot": res, "tractive_effort_graph": plot["tractive_effort_plot"], "shunting_capability_graph": plot["shunting_capability_plot"], "speed_vs_slope_table": table}
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/download_performance_report")
async def dl_perf(raw: VehiclePerformanceRawInput):
    try:
        inp, raw_inp = process_and_validate_vehicle_performance_inputs(raw)
        calc = VehiclePerformanceCalculator(inp)
        stream = calc.create_report_docx()
        return StreamingResponse(stream, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", headers={"Content-Disposition": "attachment; filename=Performance_Report.docx"})
    except Exception as e: raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
