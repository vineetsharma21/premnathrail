# =============================================================================
# BRAKING PERFORMANCE CALCULATOR
# =============================================================================
# This application calculates braking distances, forces, and stopping distances
# based on DIN EN 15746-2:2021-05 standards for railway vehicles.
#
# Features:
# - Calculate stopping distances for different gradients and speeds.
# - Compare results with EN Standard limits.
# - Generate detailed CSV reports.
# - Support for both disc and tread brakes.
# - Import and Export functionality for vehicle and track data.
# - Export PDF for using LaTeX formatting.


# =============================================================================
# IMPORTS
# =============================================================================
# --- Standard Libraries ---
import tkinter as tk  # The main library for creating the graphical user interface (GUI).
from tkinter import messagebox, ttk, filedialog  # More GUI components: pop-up messages, themed widgets, and file dialogs.
import math  # Provides mathematical functions like sine, cosine, atan, etc.
import os  # Used for interacting with the operating system, like finding file paths and opening files.
from datetime import datetime  # Used for getting the current date and time, mainly for file naming.
import csv  # Library for reading from and writing to CSV files.
import subprocess  # Allows running external commands, used here to run 'pdflatex' for PDF generation.
import re  # Regular expressions library, used for finding and replacing text (e.g., escaping LaTeX characters).
import shutil  # Used for high-level file operations like moving files across different drives.
from jinja2 import Environment, FileSystemLoader  # A templating engine used to fill the LaTeX template with data.
from PIL import Image, ImageTk  # Pillow library, used for handling and displaying images (like logos and brake photos) in the GUI.

# =============================================================================
# CORE CALCULATION LOGIC & EVENT HANDLERS
# =============================================================================
# The functions and classes below are moved here to be defined before they are called.

# --- Tooltip Class ---
class Tooltip:
    def __init__(self, widget, text):# Create a tooltip for a widget.
        self.widget = widget # The widget to which the tooltip is attached.
        self.text = text# The text to display in the tooltip.
        self.tooltip_window = None# The window that will display the tooltip.
        self.widget.bind("<Enter>", self.show_tooltip)# Show the tooltip when the mouse enters the widget.
        self.widget.bind("<Leave>", self.hide_tooltip)# Hide the tooltip when the mouse leaves the widget.

    def show_tooltip(self, event=None):# Show the tooltip.
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25# Position the tooltip slightly offset from the cursor
        y += self.widget.winfo_rooty() + 25# Position the tooltip slightly offset from the cursor
        
        # Create a Toplevel window for the tooltip
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True) # Remove window decorations
        self.tooltip_window.wm_geometry(f"+{x}+{y}")# Position the tooltip
        
        label = tk.Label(self.tooltip_window, text=self.text, background="yellow", relief="solid", borderwidth=1,# Background color and border style)
                         font=("Helvetica", 8))
        label.pack(padx=1, pady=1)

    def hide_tooltip(self, event=None):# Hide the tooltip.
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None

# --- ZoomableImage Class ---
class ZoomableImage:
    def __init__(self, parent, image_path, title):# Create a zoomable image window.
        self.parent = parent
        self.image_path = image_path
        self.title = title
        self.toplevel = tk.Toplevel(parent)
        self.toplevel.title(title)
        self.toplevel.resizable(False, False)
        
        self.img_label = tk.Label(self.toplevel, relief="solid", borderwidth=1, bg="#ffffff")
        self.img_label.pack(padx=10, pady=10)
        self.img_label.bind("<Double-Button-1>", self.toggle_zoom)
        
        self.is_zoomed = False
        self.original_image = self.load_image(150, 150)
        self.zoomed_image = self.load_image(400, 400)
        self.set_image(self.zoomed_image)
        self.is_zoomed = True

    def load_image(self, width, height):# Load an image and resize it to the specified width and height.
        try:
            img = Image.open(self.image_path)
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(img)
        except (FileNotFoundError, AttributeError):
            placeholder = Image.new('RGB', (width, height), color='#ffffff')
            return ImageTk.PhotoImage(placeholder)
        except Exception as e:
            messagebox.showerror("Image Error", f"Could not load image.\nError: {e}")
            return None

    def set_image(self, photo_image):# Set the image in the label.
        self.img_label.config(image=photo_image)
        self.img_label.image = photo_image # Keep a reference

    def toggle_zoom(self, event=None):# Toggle between zoomed and original image sizes.
        if self.is_zoomed:
            self.set_image(self.original_image)
            self.is_zoomed = False
        else:
            self.set_image(self.zoomed_image)
            self.is_zoomed = True

def on_image_double_click(event):
    """Handles double-click on the image to open a zoomed view."""
    # Get the current image path and title from the global map
    image_map = {
        entry_gradient: ("gradient.jpg", "Gradient"),
        entry_max_curve: ("curve.jpg", "Curve"),
        entry_superelevation: ("superelevation.png", "Superelevation"),
        entry_cant: ("cant.png", "Cant"),
        entry_gauge: ("gauge.jpg", "Gauge"),
    }
    focused_widget = root.focus_get()
    image_name, image_title = image_map.get(focused_widget, ("gradient.jpg", "Gradient"))
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir,image_name)
    
    # Check if a zoom window is already open and close it.
    if hasattr(on_image_double_click, 'zoom_window') and on_image_double_click.zoom_window:# Check if a zoom window is already open.
        on_image_double_click.zoom_window.toplevel.destroy()# Close the existing zoom window.
        on_image_double_click.zoom_window = None
    elif os.path.exists(image_path):
        # Create a new zoom window only if the image file exists.
        on_image_double_click.zoom_window = ZoomableImage(root, image_path, image_title)# Create a new zoom window.
    else:
        # If the file is missing, show an error message instead of a blank window.
        messagebox.showerror("Image Not Found", f"Cannot zoom. Image '{image_name}' not available.")


# =============================================================================
# DATA DEFINITIONS (Hardcoded Standards)
# =============================================================================
# All data is based on DIN EN 15746-2:2021-05 standard for railway applications.

# Standard braking distances for different speeds.
# Format: {speed_kmh: braking_distance_meters}
braking_data = {
    8: 3, 10: 5, 16: 12, 20: 20, 24: 28,
    30: 45, 32: 50, 40: 75, 50: 135, 60: 180
}

# Maximum allowed total stopping distances for different speeds.
# Format: {speed_kmh: max_allowed_distance_meters}
max_stopping_distances = {
    8: 6, 10: 9, 16: 18, 20: 27, 24: 36,
    30: 55, 32: 60, 40: 90, 50: 155, 60: 230,
    70: 300, 80: 400, 90: 500, 100: 620
}

# Standard gravitational acceleration constant in m/s².
g = 9.81

# Standard reaction time for the driver in seconds.
reaction_time = 1.0
# A global dictionary to store the placeholder values for the PDF
pdf_report_placeholders = {}

# =============================================================================
# CORE CALCULATION LOGIC & EVENT HANDLERS 
# =============================================================================
def prepare_calculation_data(inputs):# create a function and pass inputs dictionary.
    data = inputs.copy()# inputs to copy and save the original data for report
    data['v_ms'] = round(data['speed_kmh'] * (1000 / 3600), 2)# convert speed from km/h to m/s and round to 2 decimal places and store in the dictionary.
    gradient_type = data['gradient_type']# Get the gradient type from the input data.
    gradient_input = data['gradient_input']
    if gradient_type == "Degree (°)":# If the gradient type is in degrees, use it directly.
        angle_deg = gradient_input
    elif gradient_type == "1 in G":# If the gradient type is in "1 in G" format, convert it to degrees and round to 2 decimal places.
        angle_deg = round(math.degrees(math.atan(1 / gradient_input)), 2) if gradient_input != 0 else 0# change 1 in G to radian and then convert to degrees.from this formula angle = atan(1/G). Handle division by zero.
    elif gradient_type == "Percentage (%)":# If the gradient type is in percentage, convert it to degrees and round to 2 decimal places.
        angle_deg = round(math.degrees(math.atan(gradient_input / 100)), 2)# change percentage to radian and then convert to degrees. and avoid division by zero.
    else:# Invalid gradient type, raise an error.
        raise ValueError("Invalid gradient type provided.")
    data['angle_deg'] = angle_deg# Store the calculated angle in the dictionary.
    data['reaction_time'] = reaction_time# Store the standard reaction time in the dictionary.
    return data# Return the dictionary with all the prepared data.

def perform_calculations(data):# Function to perform the main calculations based on the prepared data.
    results = {}# Initialize an empty dictionary to store the results.
    mass_kg = data['mass_kg']# Get the mass of the vehicle in kg from the input data.
    old_data_for_report = {}# Initialize an empty dictionary to store old data for the report.
    for speed, braking_distance_val in sorted(braking_data.items()):# Loop through each speed and its corresponding braking distance from the standard data.
        if speed > round(data['speed_kmh']):# Only calculate for speeds up to the input speed.
            continue# Skip speeds greater than the input speed.
        vi_old = round(speed * (1000 / 3600), 2)# Convert the speed from km/h to m/s and round to 2 decimal places. from this formula vi = v * (1000/3600).
        # Avoid division by zero
        deceleration_old = round((0 - vi_old ** 2) / (2 * braking_distance_val), 4) if braking_distance_val != 0 else 0.0# Calculate the deceleration using the formula a = (0 - vi^2) / (2 * d) and round to 4 decimal places. Handle division by zero.
        braking_force_old = round(mass_kg * abs(deceleration_old), 2)# Calculate the braking force using F = m * a and round to 2 decimal places.
        reaction_distance_old = round(vi_old * data['reaction_time'], 2)# Calculate the reaction distance using d = v * t and round to 2 decimal places.
        total_stopping_distance_old = round(reaction_distance_old + braking_distance_val, 2)# Calculate the total stopping distance using d + d_reaction and round to 2 decimal places.
        old_data_for_report[speed] = {#store all the calculated values in the dictionary in speed as key.
            'speed_ms': vi_old, 'braking_distance': braking_distance_val, 'deceleration': abs(deceleration_old),# store all the calculated values in the dictionary in speed as key.
            'reaction_distance': reaction_distance_old, 'total_stopping_distance': total_stopping_distance_old,# store all the calculated values in the dictionary in speed as key.
            'braking_force': braking_force_old# store all the calculated values in the dictionary in speed as key.
        }
    results['old_data_for_report'] = old_data_for_report# Store the old data for the report in the results dictionary.

    if data.get('distance_source') == 'Custom' and data.get('custom_distance') and data['custom_distance'] > 0:# If the user has selected "Custom" distance source and provided a valid custom distance.
        total_stopping_distance = data['custom_distance']
        reaction_distance = data['v_ms'] * data['reaction_time']
        braking_distance = total_stopping_distance - reaction_distance
        # This is net deceleration required on a flat surface
        deceleration_custom = (data['v_ms']**2) / (2 * braking_distance) if braking_distance > 0 else 0.0
        # This is the net braking force required on a flat surface
        braking_force_custom = mass_kg * deceleration_custom
        results['f_b'] = round(braking_force_custom, 2)# Store the calculated braking force in the results dictionary.
        results['braking_force_source'] = "Calculated (from Custom Inputs)"# Store the source of the braking force in the results dictionary.
    else:# Otherwise, use the maximum braking force from the standard data.
        max_braking_force = max(d['braking_force'] for d in old_data_for_report.values()) if old_data_for_report else 0.0# Get the maximum braking force from the old data.
        results['f_b'] = round(max_braking_force, 2)# Store the maximum braking force in the results dictionary.
        results['braking_force_source'] = "Calculated (from EN Standards)"# Store the source of the braking force in the results dictionary.

    return results# Return the results dictionary.
def convert():
    """
    Main function that populates the GUI table and simultaneously captures
    the specific data needed for the PDF report from the main calculation loop.
    """
    # Use the global variable to store placeholder data for the PDF
    global pdf_report_placeholders# create a global variable to store data for the PDF report
    pdf_report_placeholders.clear() # Clear any data from previous calculations

    pdf_data = _get_and_validate_inputs()# Get and validate user inputs from the GUI
    if not pdf_data:# If validation fails, exit the function
        return# Exit if validation fails
    
    # Initialize lists to store all speed/gradient combination results
    pdf_report_placeholders['rail_results'] = []  # List of dicts for rail mode results
    pdf_report_placeholders['road_results'] = []  # List of dicts for road mode results
    pdf_report_placeholders['speed_list'] = []  # List of speeds for table headers
    pdf_report_placeholders['gradient_list'] = []  # List of gradients for table rows
        
    try:# Try block to catch any calculation errors
        # Clear old results from the GUI table
        for i in tree.get_children():# Loop through each item in the treeview
            tree.delete(i)# Delete the item
            
        # Prepare initial data and get the max braking force
        initial_prepared_data = prepare_calculation_data(pdf_data)# Prepare the data for calculations
        
        # Mode-dependent setup for braking force and initial placeholders
        max_braking_force = 0
        if distance_source_var.get() == "EN Standard":
            initial_results = perform_calculations(initial_prepared_data)# This function is now simpler
            max_braking_force = initial_results['f_b']# Get the max braking force from the results
            pdf_report_placeholders['old_data_for_report'] = initial_results.get('old_data_for_report', {})# store old data for the PDF report
        else: # Custom Mode
            pdf_report_placeholders['old_data_for_report'] = {}
            # In Custom mode, calculate the required braking force for the max speed
            # to populate the PDF report's summary calculations.
            if pdf_data.get('custom_distance') and pdf_data['custom_distance'] > 0:
                v_ms = initial_prepared_data['v_ms']
                custom_dist = pdf_data['custom_distance']
                mass_kg_local = pdf_data['mass_kg']

                # 1. Compute reaction distance at max speed
                reaction_dist_at_max = v_ms * reaction_time
                
                # 2. Compute braking distance required to meet the custom total distance
                braking_dist_at_max = custom_dist - reaction_dist_at_max

                if braking_dist_at_max > 0:
                    # 3. Compute deceleration required at max speed
                    decel_at_max = (v_ms**2) / (2 * braking_dist_at_max)
                    
                    # 4. Compute the equivalent braking force on a flat surface
                    braking_force_at_max = mass_kg_local * decel_at_max
                    
                    # Set max_braking_force so it can be used in the placeholder update below
                    max_braking_force = braking_force_at_max
                    
                    # 5. Store all calculated values for the PDF report summary
                    pdf_report_placeholders['fb'] = round(braking_force_at_max, 2)
                    pdf_report_placeholders['decel'] = round(decel_at_max, 2)
                    pdf_report_placeholders['braking_distance_flat'] = round(braking_dist_at_max, 2)
                    pdf_report_placeholders['total_stopping_distance_for_flat'] = round(custom_dist, 2)
                else:
                    max_braking_force = 0 # Cannot calculate force if braking distance is not positive

        mass_kg = pdf_data['mass_kg']# Get the mass of the vehicle in kg from the input data
        
        # Store common values needed for the PDF report
        pdf_report_placeholders.update({# store common values needed for the PDF report
            'doc_no': escape_latex(entry_doc_no.get()),# Escape LaTeX special characters in document number
            'made_by': escape_latex(entry_made_by.get()),
            'checked_by': escape_latex(entry_checked_by.get()),
            'approved_by': escape_latex(entry_approved_by.get()),
            'mass_kg': mass_kg,# store mass in kg
            'weight_n': round(mass_kg * g, 2),# store weight in Newtons
            'speed_kmh': pdf_data['speed_kmh'],# store speed in km/h
            'reaction_time': reaction_time ,# store reaction time
            'v_ms': initial_prepared_data['v_ms'],# store speed in m/s
            'v_ms_ms': round(initial_prepared_data['v_ms']**2, 2),# store speed squared in m²/s²
            'number_of_wheels': pdf_data['number_of_wheels'],# store number of wheels
            'wheel_dia': pdf_data['wheel_dia'] if pdf_data['wheel_dia'] else 0,# store wheel diameter
            'wheel_radius': pdf_data['wheel_radius'] if pdf_data['wheel_radius'] else 0,# store wheel radius
            'max_braking_force': round(max_braking_force, 2) if max_braking_force > 0 else "N/A (Custom)",
            'min_braking_force': round(max_braking_force / pdf_data['number_of_wheels'], 2) if pdf_data['number_of_wheels'] > 0 and max_braking_force > 0 else "N/A (Custom)",
            'Reaction_distance': round(initial_prepared_data['v_ms'] * reaction_time, 2),#  store reaction distance
            'gradient_input': pdf_data['gradient_input'],# store gradient input
            'gradient_type':  escape_latex(pdf_data['gradient_type']),# Escape LaTeX special characters in gradient type
            'angle_deg': initial_prepared_data['angle_deg'],# store angle in degrees
            'reference_speed_for_force': pdf_data['speed_kmh'],# store reference speed for force calculation
            'reference_braking_dist': braking_data.get(pdf_data['speed_kmh'], 0),#   store reference braking distance
            'totl_sto_distan': round( (initial_prepared_data['v_ms'] * reaction_time) + braking_data.get(pdf_data['speed_kmh'], 0) , 2),# store total stopping distance
            'friction_coefficient': pdf_data.get('friction', 0.7),# store friction coefficient
            # Initialize road mode placeholders
            'road_angle_deg_flat': 0,
            'road_fg_flat': 0,
            'road_fnet_flat': 0,
            'road_deceleration_flat': 0,
            'road_braking_distance_flat': 0,
            'road_total_stop_flat': 0,
            'road_angle_deg_max': 0,
            'road_fg_max_gradient': 0,
            'road_fnet_max_gradient': 0,
            'road_deceleration_max_gradient': 0,
            'road_braking_distance_max_gradient': 0,
            'road_total_stop_max_gradient': 0,
            # Initialize GBR placeholder
            'gbr': 0
        })

        # Loop parameters
        max_gradient = pdf_data['gradient_input']#  Get the maximum gradient from the input data
        num_gradient_steps = pdf_data['gradient_increment']# Get the number of gradient steps from the input data
        gradient_step = max_gradient / num_gradient_steps if num_gradient_steps > 0 else 0# Calculate the gradient step size
        max_speed = pdf_data['speed_kmh']# Get the maximum speed from the input data
        speed_increment = pdf_data['speed_increment']# Get the speed increment from the input data

        def format_val(val):# Helper function to format values for display
            return  f"{val:.2f}"# Format the value for display

        def run_calculations_for_gradient(gradient_value):# Function to run calculations for a specific gradient
            is_max_gradient = (gradient_value == max_gradient) # Check if this is the target gradient for the PDF

            gradient_header_text = f"Gradient: {gradient_value:.2f}"# Generate the gradient header text
            tree.insert("", "end", values=(gradient_header_text, "", "", "", "", "", "", "", "", ""), tags=('gradient_header',))# Insert the gradient header into the GUI table

            temp_gradient_data = {'gradient_input': gradient_value, 'gradient_type': pdf_data['gradient_type']}# Temporary dictionary to hold gradient data for angle calculation
            if temp_gradient_data['gradient_type'] == "Degree (°)":# If the gradient type is in degrees, use it directly.
                angle_deg = temp_gradient_data['gradient_input']# Use the input directly
            elif temp_gradient_data['gradient_type'] == "1 in G":# If the gradient type is in "1 in G" format, convert it to degrees and round to 2 decimal places.
                angle_deg = round(math.degrees(math.atan(1 / temp_gradient_data['gradient_input'])),2) if temp_gradient_data['gradient_input'] != 0 else 0# Handle division by zero.
            else:# If the gradient type is neither in degrees nor in "1 in G", use a default value of 0 degrees.
                angle_deg = round(math.degrees(math.atan(temp_gradient_data['gradient_input'] / 100)),2)# Convert percentage to degrees and round to 2 decimal places.
          
            gravitational_force_fg = round(mass_kg * g * (math.sin(math.radians(angle_deg))), 2)# Calculate the gravitational force component along the slope and round to 2 decimal places.
            
            scenario_list = []
            if gradient_value == 0:# Flat surface scenario
                scenario_list.append("Straight Track")
            else:
                scenario_list.append("Moving up")# Moving up scenario                                         
                scenario_list.append("Moving down")# Moving down scenario

            for scenario_name in scenario_list:# Loop through each scenario and its corresponding net force
                tree.insert("", "end", values=(scenario_name, "", "", "", "", "", "", "", "", ""), tags=('direction_header',))# Insert the scenario header into the GUI table

                # Start the loop from the first increment, not from 0.
                current_speed = speed_increment
                
                while current_speed <= max_speed:# Loop until the current speed exceeds the maximum speed
                    is_max_speed = (current_speed == max_speed) # Check if this is the target speed for the PDF

                    current_speed_kmh = current_speed# Current speed in km/h
                    v_ms = current_speed_kmh * (1000 / 3600)# Convert current speed to m/s
                    reaction_distance = round(v_ms * reaction_time, 2)# Calculate reaction distance
                    row_values = () # Initialize tuple

                    # --- DIVERGENT CALCULATION LOGIC ---
                    if distance_source_var.get() == 'Custom' and pdf_data.get('custom_distance'):
                        # --- Custom Distance: Calculate required forces/deceleration in reverse ---
                        total_stopping_distance = pdf_data['custom_distance'] # Use exact, unrounded value
                        braking_distance = total_stopping_distance - reaction_distance
                        
                        a_deceleration = (v_ms**2) / (2 * braking_distance) if braking_distance > 0 else 0.0#  Calculate deceleration
                        f_net = mass_kg * a_deceleration

                        # Calculate the required applied force to achieve the net force
                        if scenario_name == "Moving up": # Gravity helps braking
                            applied_force = f_net - gravitational_force_fg
                        elif scenario_name == "Moving down": # Gravity works against braking
                            applied_force = f_net + gravitational_force_fg
                        else: # Straight Track
                            applied_force = f_net
                        
                        applied_force = (applied_force) # Applied force can't be negative
                        compliance_status = "" # No compliance check for custom mode
                        
                        row_values = (# Prepare the row values for insertion into the GUI table
                            f"{current_speed_kmh:.2f}", f"{v_ms:.2f}", f"{reaction_distance:.2f}",# Speed, speed in m/s, reaction distance
                            f"{applied_force:.2f}",
                            f"{gravitational_force_fg:.2f}",
                            f"{f_net:.2f}",
                            f"{a_deceleration:.2f}",#    Deceleration, braking distance, total stopping distance
                            format_val(braking_distance),
                            str(total_stopping_distance), # Show exact user input
                            compliance_status# Compliance status
                        )

                    else:
                        # --- EN Standard: Calculate distances from a fixed braking force ---
                        if scenario_name == "Moving up":
                            f_net = max_braking_force + gravitational_force_fg
                        elif scenario_name == "Moving down":
                            f_net = max_braking_force - gravitational_force_fg
                        else: # Straight Track
                            f_net = max_braking_force
                        
                        a_deceleration = abs(f_net / mass_kg) if mass_kg > 0 else 0.0#  Calculate deceleration
                        
                        # Only calculate braking distance if there is a deceleration value
                        if a_deceleration > 0 and v_ms > 0:
                            braking_distance = abs((0 - v_ms**2) / (2 * a_deceleration))# Calculate braking distance
                        else:
                            braking_distance = 0.0
                        
                        total_stopping_distance = reaction_distance + braking_distance# Calculate total stopping distance
                        compliance_status = get_standard_compliance(current_speed_kmh, total_stopping_distance)# Check compliance with EN standard

                        row_values = (# Prepare the row values for insertion into the GUI table
                            f"{current_speed_kmh:.2f}", f"{v_ms:.2f}", f"{reaction_distance:.2f}",# Speed, speed in m/s, reaction distance
                            f"{max_braking_force:.2f}", f"{gravitational_force_fg:.2f}", f"{f_net:.2f}",# Braking force, gravitational force, net force
                            f"{a_deceleration:.2f}",#    Deceleration, braking distance, total stopping distance
                            format_val(braking_distance),
                            format_val(total_stopping_distance),
                            compliance_status# Compliance status
                        )

                    tree.insert("", "end", values=row_values, tags=('evenrow' if int(current_speed_kmh) % 2 == 0 else 'oddrow',))# Insert the row into the GUI table with alternating row colors
                    
                    # Store ALL results for LaTeX table generation
                    pdf_report_placeholders['rail_results'].append({
                        'speed_kmh': round(current_speed_kmh, 2),
                        'gradient': round(gradient_value, 2),
                        'scenario': scenario_name,
                        'v_ms': round(v_ms, 2),
                        'reaction_distance': round(reaction_distance, 2),
                        'applied_force': row_values[3],
                        'gravitational_force': round(gravitational_force_fg, 2),
                        'net_force': row_values[5],
                        'deceleration': row_values[6],
                        'braking_distance': row_values[7],
                        'total_stopping_distance': row_values[8],
                        'compliance': row_values[9]
                    })
                    
                    # --- THIS IS THE NEW LOGIC TO CAPTURE PDF DATA ---
                    # If this is the max speed and max gradient, save the values
                    if is_max_speed and is_max_gradient:# If this is the target speed and gradient for the PDF
                        if scenario_name == "Moving up":# Moving up scenario
                            pdf_report_placeholders['fmax'] = round(pdf_report_placeholders['weight_n'] * (math.sin(math.radians(angle_deg))),2)
                            pdf_report_placeholders['f_g'] = round(gravitational_force_fg,2)# Gravitational force
                            pdf_report_placeholders['f_net_Moving_up'] = round(f_net, 2)# Net force
                            pdf_report_placeholders['a_deceleration_new_Moving_up'] = round(a_deceleration, 2)# Deceleration
                            pdf_report_placeholders['braking_distance_d_newMoving_up'] = round(braking_distance, 2)# Braking distance
                            pdf_report_placeholders['total_stopping_distance_ts_new_Moving_up'] = round(total_stopping_distance, 2)# Total stopping distance
                            pdf_report_placeholders['a_decele_up'] = round(2 * a_deceleration, 2)# deceleration square for report
                        elif scenario_name == "Moving down":# Moving down scenario
                            pdf_report_placeholders['fmax'] = round(pdf_report_placeholders['weight_n'] * (math.sin(math.radians(angle_deg))),2)# Max gravitational force
                            pdf_report_placeholders['f_net_Moving_down'] = round(f_net, 2)# Net force
                            pdf_report_placeholders['a_deceleration_new_Moving_down'] = round(a_deceleration, 2)# Deceleration
                            pdf_report_placeholders['braking_distance_d_new_Moving_down'] = round(braking_distance, 2)# Braking distance
                            pdf_report_placeholders['total_stopping_distance_ts_new__Moving_down'] = round(total_stopping_distance, 2)# Total stopping distance for report's intro calculation
                            pdf_report_placeholders['a_decele_down'] = round(2 * a_deceleration, 2)# deceleration square for report
                    
                    # Also capture data for the "Flat Surface" (Gradient 0) scenario at max speed
                    if is_max_speed and gradient_value == 0:# If this is the target speed and gradient for the PDF at max speed
                        pdf_report_placeholders['fmax'] = round(pdf_report_placeholders['weight_n'] * 0)# Max gravitational force is 0 on flat surface
                        pdf_report_placeholders['f_g_flat'] = 0# Gravitational force is 0 on flat surface
                        pdf_report_placeholders['f_net_flat'] = round(f_net, 2)# Net force
                        pdf_report_placeholders['a_deceleration_flat'] = round(a_deceleration, 2)# Deceleration
                        pdf_report_placeholders['braking_distance_flat'] = round(braking_distance, 2)# Braking distance
                        pdf_report_placeholders['total_stopping_distance_for_flat'] = round(total_stopping_distance, 2)# Total stopping distance
                        pdf_report_placeholders['decele_flat_d'] = round(2 * a_deceleration, 2)# deceleration square for report
                        # Also get the deceleration for the report's intro calculation
                        if braking_data.get(max_speed, 0) > 0:# If the braking distance is available for this speed
                           pdf_report_placeholders['decel'] = round(abs((0-v_ms**2) / (2 * braking_data[max_speed])), 2)# Calculate deceleration
                           pdf_report_placeholders['fb'] = round(mass_kg * pdf_report_placeholders['decel'],2)# Calculate braking force
                        else:# If not available, set to 0
                           pdf_report_placeholders['decel'] = 0# Set to 0
            
                    # Increment logic
                    current_speed += speed_increment# Increment the current speed by the speed increment
                    if current_speed > max_speed and current_speed - speed_increment < max_speed:
                        current_speed = max_speed
            
            tree.insert("", "end", values=("", "", "", "", "", "", "", "", "", ""))# Insert a blank row for better readability
        
        run_calculations_for_gradient(0.0)# Always run calculations for flat surface (0% gradient)

        for j in range(1, num_gradient_steps + 1):# Loop through each gradient step
            current_gradient = j * gradient_step# Calculate the current gradient
            if current_gradient > 0:# Only run for positive gradients
                run_calculations_for_gradient(current_gradient)#    Run calculations for the current gradient

        # ---------------------------------------------------------------------
        # ----- ROAD MODE (Friction Based) ------------------------------------
        # ---------------------------------------------------------------------
        # This second block reuses the same gradient and speed loop logic.
        # Only run if the checkbox is enabled
        if road_mode_enabled_var.get():
            tree.insert("", "end", values=("----- ROAD MODE (Friction Based) -----", "", "", "", "", "", "", "", "", ""), tags=("gradient_header",))

            friction = pdf_data.get('friction', 0.7)
            normal_force = mass_kg * g
            fb_friction = friction * normal_force

            def run_road_mode_for_gradient(gradient_value):
                # Gradient header row
                tree.insert("", "end", values=(f"Road Gradient: {format_val(gradient_value)}%", "", "", "", "", "", "", "", "", ""), tags=("gradient_header",))
                row_index = 0
                is_max_gradient_road = (gradient_value == max_gradient)
                
                current_speed = speed_increment
                while current_speed <= max_speed:
                    v_ms = round(current_speed * (1000 / 3600), 2)
                    reaction_distance = round(v_ms * reaction_time, 2)

                    # Angle conversion based on selected gradient type
                    angle_deg = 0.0
                    if gradient_type_var.get() == "Degree (°)":
                        angle_deg = gradient_value
                    elif gradient_type_var.get() == "1 in G":
                        if gradient_value != 0:
                            try:
                                angle_deg = round(math.degrees(math.atan(1 / gradient_value)), 2)
                            except Exception:
                                angle_deg = 0.0
                    elif gradient_type_var.get() == "Percentage (%)":
                        try:
                            angle_deg = round(math.degrees(math.atan(gradient_value / 100)), 2)
                        except Exception:
                            angle_deg = 0.0

                    theta_rad = math.radians(angle_deg)
                    f_g = round(mass_kg * g * math.sin(theta_rad), 2)
                    f_net = round(fb_friction - f_g, 2)
                    a = round(f_net / mass_kg, 6) if mass_kg != 0 else 0.0

                    # Validation: skip non-physical cases
                    if a <= 0 or v_ms <= 0:
                        continue

                    try:
                        braking_distance = round((v_ms ** 2) / (2 * a), 2) if a > 0 else 0.0
                    except Exception:
                        braking_distance = float('inf')

                    if braking_distance <= 0 or braking_distance == float('inf'):
                        continue

                    total_stopping_distance = round(reaction_distance + braking_distance, 2)

                    # Row formatting & insertion
                    tag = "evenrow" if (row_index % 2 == 0) else "oddrow"
                    tree.insert(
                        "", "end",
                        values=(
                            round(current_speed, 2),
                            v_ms,
                            reaction_distance,
                            round(fb_friction, 2),
                            f_g,
                            f_net,
                            round(a, 4),
                            braking_distance,
                            total_stopping_distance,
                            ""  # No EN compliance for road mode
                        ),
                        tags=(tag,)
                    )
                    row_index += 1
                    
                    # Store ALL road mode results for LaTeX table generation
                    pdf_report_placeholders['road_results'].append({
                        'speed_kmh': round(current_speed, 2),
                        'gradient': round(gradient_value, 2),
                        'v_ms': v_ms,
                        'reaction_distance': reaction_distance,
                        'friction_force': round(fb_friction, 2),
                        'gravitational_force': f_g,
                        'net_force': f_net,
                        'deceleration': round(a, 4),
                        'braking_distance': braking_distance,
                        'total_stopping_distance': total_stopping_distance,
                        'angle_deg': round(angle_deg, 2)
                    })

                    # Capture data for PDF report at max speed
                    is_max_speed_road = (current_speed == max_speed)
                    if is_max_speed_road and gradient_value == 0:
                        # Road mode flat surface at max speed
                        pdf_report_placeholders['road_friction'] = friction
                        pdf_report_placeholders['road_normal_force'] = round(normal_force, 2)
                        pdf_report_placeholders['road_fb_friction_flat'] = round(fb_friction, 2)
                        pdf_report_placeholders['road_fg_flat'] = round(f_g, 2)
                        pdf_report_placeholders['road_fnet_flat'] = round(f_net, 2)
                        pdf_report_placeholders['road_deceleration_flat'] = round(a, 4)
                        pdf_report_placeholders['road_braking_distance_flat'] = round(braking_distance, 2)
                        pdf_report_placeholders['road_total_stop_flat'] = round(total_stopping_distance, 2)
                    elif is_max_speed_road and is_max_gradient_road:
                        # Road mode at max gradient and max speed
                        pdf_report_placeholders['road_angle_deg_max'] = round(angle_deg, 2)
                        pdf_report_placeholders['road_fg_max_gradient'] = round(f_g, 2)
                        pdf_report_placeholders['road_fnet_max_gradient'] = round(f_net, 2)
                        pdf_report_placeholders['road_deceleration_max_gradient'] = round(a, 4)
                        pdf_report_placeholders['road_braking_distance_max_gradient'] = round(braking_distance, 2)
                        pdf_report_placeholders['road_total_stop_max_gradient'] = round(total_stopping_distance, 2)

                    current_speed += speed_increment
                    if current_speed > max_speed and current_speed - speed_increment < max_speed:
                        current_speed = max_speed
                        
                tree.insert("", "end", values=("", "", "", "", "", "", "", "", "", ""))

            # Run road mode for all gradient steps
            run_road_mode_for_gradient(0.0)# Always run for flat surface
            for j in range(1, num_gradient_steps + 1):
                current_gradient = j * gradient_step
                if current_gradient > 0:
                    run_road_mode_for_gradient(current_gradient)

        # --- GBR (Global Braking Ratio) calculation and display ---
        # Definition (as requested): GBR = f_b_rail / (m * g)
        # Using max_braking_force determined from rail (EN Standard or Custom) section.
        tree.insert("", "end", values=("", "", "", "", "", "", "", "", "", ""))
        if mass_kg > 0 and max_braking_force > 0:
            gbr = round(max_braking_force / (mass_kg * g), 4)
        else:
            gbr = "N/A"
        tree.insert(
            "", "end",
            values=(f"GBR (f_b / (m*g))", gbr, "", "", "", "", "", "", "", ""),
            tags=("gradient_header",)
        )
        
        # Store GBR in PDF placeholders
        pdf_report_placeholders['gbr'] = gbr if gbr != "N/A" else 0

    except Exception as e:# Catch any exceptions that occur during calculations
        messagebox.showerror("Calculation Error", f"Calculation failed.\nError: {e}")# Show an error message

# def on_widget_enter(event):
#     """Called when the mouse pointer enters a widget to update the status bar."""
#     widget = event.widget# Get the widget that triggered the event.
#     # Get the widget's class (e.g., TButton, TEntry)
#     widget_class = widget.winfo_class()# Get the class of the widget.
#     # Try to get the widget's text, if it has one
#     try:# If the widget has a text attribute, get its text. Otherwise, just show the class.
#         widget_text = widget.cget("text")# Get the text attribute of the widget.
#         info_text = f"Hovering over: {widget_class} (Text: '{widget_text}')"# If the widget has a text attribute
#     except tk.TclError:# If the widget doesn't have a text attribute, just show the class.
#         info_text = f"Hovering over: {widget_class}"# No text attribute
    
#     status_bar_var.set(info_text)# Update the status bar text

# def on_widget_leave(event):
#     """Called when the mouse pointer leaves a widget to clear the status bar."""
#     status_bar_var.set("") # Clear the status bar text
    
def validate_input(text):#""Validates numeric input for Entry widgets."""
    """Validation function to allow only numeric (float) input for an Entry widget."""
    # Allows an empty string, or a string with at most one decimal point.
    if text == "" or (text.count('.') <= 1 and text.replace('.', '', 1).isdigit()):# Allow empty input or valid float format.
        return True# Accept the input.
    return False# Reject the input.

def focus_next_widget(event):
    """Event handler to move focus to the next widget when Enter is pressed."""
    event.widget.tk_focusNext().focus()# Move focus to the next widget.
    return "break" # Prevents the default Enter key behavior.

def _get_and_validate_inputs():
    """
    Gets all user inputs from the GUI, validates them, and returns them in a dictionary.
    Returns None if validation fails.
    """
    try:# Try to convert all inputs to the appropriate types.
        # Parse comma-separated speeds
        speed_input = entry_speed.get().strip()
        if ',' in speed_input:
            speed_list = [float(s.strip()) for s in speed_input.split(',') if s.strip()]
        else:
            speed_list = [float(speed_input)] if speed_input else []
        
        # Parse comma-separated gradients
        gradient_input = entry_gradient.get().strip()
        if ',' in gradient_input:
            gradient_list = [float(g.strip()) for g in gradient_input.split(',') if g.strip()]
        else:
            gradient_list = [float(gradient_input)] if gradient_input else [0.0]
        
        pdf = {# Create a dictionary to hold all input values.
            'doc_no': escape_latex(entry_doc_no.get()),
            'made_by': entry_made_by.get(),
            'checked_by': entry_checked_by.get(),
            'approved_by': entry_approved_by.get(),
            'mass_kg': float(entry_mass.get()),
            'speed_list': speed_list,
            'speed_kmh': max(speed_list) if speed_list else 0.0,
            'number_of_wheels': int(entry_number_of_wheels.get()),
            'brake_type': brake_type_var.get(),
            'number_braked_wheels': int(entry_number_braked_wheels.get()) if entry_number_braked_wheels.get() else None,
            'wheel_dia': float(entry_wheel_dia.get()) if entry_wheel_dia.get() else None,
            'wheel_radius': float(entry_wheel_dia.get()) / 2 if entry_wheel_dia.get() else None,
            'mode': calc_mode_var.get(),
            'distance_source': distance_source_var.get(),
            'custom_distance': float(entry_distance_custom.get()) if entry_distance_custom and entry_distance_custom.get() else None,
            'clamping_force': float(entry_clamping_force.get()) if entry_clamping_force and entry_clamping_force.get() else None,
            'disc_dia': float(entry_disc_dia.get()) if entry_disc_dia and entry_disc_dia.get() else None,
            'track_standard': track_standard_var.get(),
            'gradient_input': float(entry_gradient.get()) if entry_gradient.get() else 0.0,
            'gradient_type': gradient_type_var.get(),
            'friction': float(entry_friction.get()) if entry_friction.get() else 0.7,
            'max_curve': float(entry_max_curve.get()) if entry_max_curve.get() else None,
            'max_curve_unit': max_curve_unit_var.get() if max_curve_unit_var.get() else None,
            'max_superelevation': float(entry_superelevation.get()) if entry_superelevation.get() else None,
            'max_cant': float(entry_cant.get()) if entry_cant.get() else None,
            'track_gauge': float(entry_gauge.get()) if entry_gauge.get() else None,
            'speed_increment': float(entry_speed_increment.get()) if entry_speed_increment.get() else 10.0,
            'gradient_increment': int(entry_gradient_increment.get()) if entry_gradient_increment.get() else 5,
        }
        # --- Critical Input Validation ---
        if pdf['mass_kg'] <= 0 or pdf['speed_kmh'] < 0 or pdf['number_of_wheels'] <= 0: # Ensure mass, speed, and number of wheels are positive.
            messagebox.showerror("Invalid Input", "Please enter positive numbers for weight, speed, and wheels.")
            return None

        # If user has provided a custom distance, ensure it's a positive number.
        if pdf['distance_source'] == 'Custom':
            if not pdf.get('custom_distance') or pdf['custom_distance'] <= 0:
                messagebox.showerror("Invalid Input", "Please enter a positive custom distance.")
                return None

            # --- YAHAN CHECK ADD KAREIN ---
            # Ensure the custom total stopping distance is physically possible: it must be greater
            # than the reaction distance at the specified speed.
            # Compute speed in m/s and reaction distance using the module's reaction_time (fallback to 1.0)
            speed_ms = pdf['speed_kmh'] * (1000.0 / 3600.0)
            try:
                rt = float(reaction_time)
            except Exception:
                rt = 1.0
            reaction_dist = speed_ms * rt

            if pdf['custom_distance'] <= reaction_dist:
                messagebox.showerror(
                    "Invalid Input",
                    f"Custom distance ({pdf['custom_distance']} m) must be greater than the reaction distance ({reaction_dist:.2f} m) at the specified speed.\n"
                    "Please increase the custom total stopping distance or reduce the speed."
                )
                return None
        return pdf# Return the validated input dictionary.
    except ValueError:# Handle conversion errors.
        messagebox.showerror("Input Error", "Please ensure all input fields contain valid numbers.")# Show error message.
        return None# Return None to indicate validation failure.

def get_standard_compliance(speed_kmh, calculated_distance):
    """
    Checks if the calculated stopping distance complies with the EN standard.
    """
    if calculated_distance == float('inf'):
        return "✗ Physical Limit Exceeded"

    # Find the relevant standard limit for the given speed
    reference_speed = None# The relevant speed limit for the given speed.
    allowed_distance = None# The distance that the speed limit is allowed for.
    # Reverse sorted keys taaki hum speed ke liye sahi lower bound dhoondh sakein
    for speed_limit in sorted(max_stopping_distances.keys(), reverse=True):# Loop through the speed limits in descending order.
        if speed_kmh >= speed_limit:# If the current speed is greater than or equal to the current speed limit,
            reference_speed = speed_limit# Set the reference speed to the current speed limit.
            allowed_distance = max_stopping_distances[speed_limit]# Set the allowed distance to the distance limit for the current speed limit.
            break# Break the loop once we find the relevant speed limit.
            
    if allowed_distance is None:# If no relevant speed limit is found, return "Data Not Found".
        return "Standard Not Found"

    if calculated_distance <= allowed_distance:# If the calculated distance is less than or equal to the allowed distance, return "Pass".
        return "✓ Standard Followed"
    else:# Otherwise, return "Fail".
        return "✗ Standard Exceeded"

def validate_speed_increment(new_value):
    """Validation function to keep the speed increment input between 0.1 and 50."""
    if new_value == "":# Allow empty input (user might be deleting to re-enter).
        return True
    try:# Try to convert the input to a float.
        value = float(new_value)
        # Check if the value is within the allowed range.
        return 0.1 <= value <= 50
    except ValueError:
        # Reject input if it's not a number.
        return False# Reject input if it's not a number.

def validate_gradient_increment(new_value):
    """Validation function to keep the gradient increment input between 1 and 10."""
    if new_value == "":# Allow empty input (user might be deleting to re-enter).
        return True# Allow empty input (user might be deleting to re-enter).
    try:# Try to convert the input to an integer.
        value = int(new_value)
        return 1 <= value <= 10## Check if the value is within the allowed range.
    except ValueError:# Reject input if it's not a number.
        return False

def update_track_data_ui(*args):
    """
    Auto-fills track data from a CSV file based on the radio button selection
    and enables/disables the relevant widgets.
    """
    selected_standard = track_standard_var.get()# Get the selected track standard from the radio button.

    # --- Group all related widgets for easy management ---
    entry_widgets = {# Dictionary to hold references to all Entry widgets.
        'gradient_input': entry_gradient, 'max_curve': entry_max_curve,
        'max_superelevation': entry_superelevation, 'max_cant': entry_cant,
        'track_gauge': entry_gauge
    }
    radio_vars = {# Dictionary to hold references to all radio button variables.
        'gradient_type': gradient_type_var, 'max_curve_unit': max_curve_unit_var
    }
    radio_button_frames = [grad_types_frame, curve_unit_frame]# List of frames containing radio buttons.

    # --- Reset all fields to their default (enabled and empty) state ---
    for entry in entry_widgets.values():# Enable and clear all Entry widgets.
        entry.config(state="normal")# Enable all Entry widgets.
        entry.delete(0, tk.END)# Clear all Entry widget contents.
    # Enable all radio buttons
    for frame in radio_button_frames:# Enable all radio buttons in all frames.
        for child in frame.winfo_children():# Loop through each child widget in the frame.
            if isinstance(child, ttk.Radiobutton):# Enable the radio button.
                child.config(state="normal")#  Enable the radio button.

    # If 'CUSTOM' is selected, the fields remain enabled for user input.
    if selected_standard == "CUSTOM":
        return# Exit the function early.

    # --- If a standard ('IR' or 'HSR') is selected, load data from CSV ---
    filename = f"{selected_standard}.csv"# Construct the filename based on the selected standard.
    try:# Open the CSV file and read the data.
        with open(filename, mode='r', encoding='utf-8') as infile:# Open the CSV file.
            reader = csv.reader(infile)#  Create a CSV reader object.
            next(reader) # Skip header row.
            track_data = {rows[0]: rows[1] for rows in reader}# Read the CSV data into a dictionary.

        # Populate the GUI with data from the CSV file.
        for key, widget in entry_widgets.items():# Loop through each Entry widget and populate it if data exists.
            if key in track_data:# If the key exists in the track_data dictionary, populate the widget.
                widget.insert(0, track_data[key])# Insert the data into the Entry widget.
        
        for key, var in radio_vars.items():# Loop through each radio button variable and populate it if data exists.
            if key in track_data:# If the key exists in the track_data dictionary, set the variable.
                var.set(track_data[key])# Set the radio button variable.

        # --- Lock all fields to prevent editing when a standard is selected ---
        for entry in entry_widgets.values():# Disable all Entry widgets.
            entry.config(state="readonly")# Disable all Entry widgets.
        
        for frame in radio_button_frames:# Disable all radio buttons in all frames.
            for child in frame.winfo_children():# Loop through each child widget in the frame.
                if isinstance(child, ttk.Radiobutton):# Disable the radio button.
                    child.config(state="disabled")# Disable the radio button.

    except FileNotFoundError:# Handle file not found error.
        messagebox.showerror("File Not Found", f"File '{filename}' not found. Please check.")# Show error message.
        track_standard_var.set("CUSTOM") # Revert to custom mode on error.
    except Exception as e:# Handle any other exceptions.
        messagebox.showerror("Error", f"Could not read CSV file.\nError: {e}")# Show error message.
        track_standard_var.set("CUSTOM")# Revert to custom mode on error.
        
def update_track_image(event=None):
    """Updates the track image display based on the currently focused widget."""
    # Map entry widgets to image and label text. A tuple contains (image_name, label_text).
    image_map = {
        entry_gradient: ("gradient.jpg", "Gradient"),
        entry_max_curve: ("curve.jpg", "Curve"),
        entry_superelevation: ("superelevation.png", "Superelevation"),
        entry_cant: ("cant.png", "Cant"),
        entry_gauge: ("gauge.jpg", "Gauge"),
    }
    
    # Get the currently focused widget.
    focused_widget = root.focus_get()
    
    # Get the corresponding image and label from the map.
    # Default to the gradient image if no relevant widget is focused.
    image_name, label_text = image_map.get(focused_widget, ("gradient.jpg", "Gradient"))
    
    try:
        # Construct the full path to the image file.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir,image_name)
        
        # Check if the image file exists.

        if os.path.exists(image_path):
            # Load and resize the image for display.
            img = Image.open(image_path)
            img = img.resize((200, 60), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            # Update the image label.
            track_image_label.config(image=photo, text="")
            track_image_label.image = photo # Keep a reference to prevent garbage collection.
            # Update the text label below the image.
            track_label.config(text=label_text)
        else:
            # If the image file is missing, show a placeholder.
            placeholder = Image.new('RGB', (150, 150), color='#ffffff')
            photo = ImageTk.PhotoImage(placeholder)
            track_image_label.config(image=photo, text="Image not found", font=("Helvetica", 10), compound="center")
            track_image_label.image = photo
            track_label.config(text=label_text) # Still show the label.
    except Exception as e:
        # Handle any other errors during image loading.
        messagebox.showerror("Image Error", f"Could not load image.\nError: {e}")

def populate_standard_table_display():# Function to populate the standard stopping distance table in the Text widget.
    """Populates the Text widget with the formatted standard stopping distance table."""
    try:
        # Temporarily enable the widget to modify its content.
        standard_text_display.config(state="normal")# Make the Text widget editable.
        standard_text_display.delete("1.0", tk.END) # Clear old content.

        # --- Configure tags for text styling (alignment, font) ---
        standard_text_display.tag_configure("right", justify="right")# Right-align text.
        standard_text_display.tag_configure("center_bold", justify="center", font=("Helvetica", 10, "bold"))# Center-align and bold text.
        standard_text_display.tag_configure("table_font", font=("Courier New", 10)) # Monospaced font for table alignment.

        # --- Insert header text ---
        standard_text_display.insert(tk.END, "DIN EN 15746-2:2021-05\n", "right")# Right-align text.
        standard_text_display.insert(tk.END, "EN 15746-2:2020(E)\n\n", "right")# Right-align text.
        standard_text_display.insert(tk.END, "Table 4 - Stopping distance\n\n", "center_bold")# Center-align and bold text.

        # --- Manually create the complex table header ---
        header_line1 = "Machine| Maximum stopping distance on\n"# First column width 
        header_line2 = "Speed  | Straight Track of machine and any\n"# Second column 
        header_line3 = "       | permitted (by the manufacturer)\n"# Third column 
        header_line4 = "       | unbraked trailing load\n"# Fourth column
        header_line5 = "(km/h) | (m)\n"# Fifth column
        separator = "+" + "-"*6 + "+" + "-"*31 + "+\n"# Separator line with 6 and 31 dashes.
        
        # Insert the header and separator lines into the widget.
        standard_text_display.insert(tk.END, separator, "table_font")# Insert the table header and separator.
        standard_text_display.insert(tk.END, header_line1, "table_font")# Insert the table header and separator.
        standard_text_display.insert(tk.END, header_line2, "table_font")# Insert the table header and separator.
        standard_text_display.insert(tk.END, header_line3, "table_font")# Insert the table header and separator.
        standard_text_display.insert(tk.END, header_line4, "table_font")# Insert the table header and separator.
        standard_text_display.insert(tk.END, header_line5, "table_font")# Insert the table header and separator.
        standard_text_display.insert(tk.END, separator, "table_font")# Insert the table header and separator.

        # --- Insert data rows from the `max_stopping_distances` dictionary ---
        for speed, distance in max_stopping_distances.items():# Loop through each speed and its corresponding max stopping distance.
            # Use f-string formatting for precise alignment.
            row_text = f"| {speed:<5}| {distance:<30}|\n"# Use f-string formatting for precise alignment.
            standard_text_display.insert(tk.END, row_text, "table_font")# Insert the row data for the table.
        
        # Add the bottom border of the table.
        standard_text_display.insert(tk.END, separator, "table_font")

        # Make the Text widget read-only again.
        standard_text_display.config(state="disabled")#

    except Exception as e:
        # Handle any unexpected errors during table population.
        standard_text_display.config(state="normal")# Make the Text widget editable.
        standard_text_display.delete("1.0", tk.END)# Clear old content.
        standard_text_display.insert(tk.END, f"Error populating table: {e}")# Display error message.
        standard_text_display.config(state="disabled")# Make the Text widget read-only again.

def update_brake_image(*args):
    """Updates the brake image display based on the selected brake type."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))# Get the directory of the current script.
        image_map = {"Disc Brake": "disc_brake.jpg", "Tread Brake": "tread_brake.png"}# Map brake types to image filenames.
        image_path = os.path.join(script_dir, image_map.get(brake_type_var.get()))# Get the image path based on the selected brake type.
        
        if os.path.exists(image_path):# Check if the image file exists.
            img = Image.open(image_path)# Open the image file.
            # Resize image with high-quality downsampling.
            img = img.resize((100, 100), Image.Resampling.LANCZOS)# Resize the image.
            photo = ImageTk.PhotoImage(img)# Convert the image to a PhotoImage.
            brake_image_small_label.config(image=photo)# Update the label with the new image.
            brake_image_small_label.image = photo # Keep a reference to avoid garbage collection.
        else:
            # If the image is not found, display a placeholder.
            placeholder = Image.new('RGB', (100, 100), color='#ffffff')# Create a blank white image.
            photo = ImageTk.PhotoImage(placeholder)# Convert to PhotoImage.
            brake_image_small_label.config(image=photo, text="Image not found", font=("Helvetica", 8), compound="center")# Update label.
            brake_image_small_label.image = photo# Keep reference.
    except Exception as e:# Handle any unexpected errors during image loading.
        messagebox.showerror("Image Error", f"Could not load image.\nError: {e}")# Display error message.

def update_calc_mode_ui(*args):
    """Dynamically updates the 'Calculate' section UI when the mode is changed."""
    # These globals are needed to store references to the dynamically created Entry widgets.
    global entry_distance_custom, entry_clamping_force, entry_disc_dia

    # Clear any widgets currently in the frame.
    for w in calc_content_frame.winfo_children():# Loop through all child widgets in the calculation content frame.
        w.destroy()# Destroy the widgets.

    mode = calc_mode_var.get()# Get the selected mode.
    
    # --- Create UI for "Braking force" mode ---
    if mode == "force":# If the selected mode is "force",
        ttk.Label(calc_content_frame, text="Total Stop Dist (m):").grid(row=0, column=0, sticky="e", padx=2, pady=2)# Label for distance input.
        entry_distance_custom = ttk.Entry(calc_content_frame, width=10, validate="key", validatecommand=(vcmd, '%P'))# Create Entry widget for distance input.
        entry_distance_custom.grid(row=0, column=1, sticky="w", padx=2, pady=2, ipadx=2, ipady=2)# Place the Entry widget in the grid.
        entry_distance_custom.bind("<Return>", focus_next_widget)# Bind the Enter key to trigger the focus_next_widget function.
        
        # Frame for the "Custom" vs "EN Standard" radio buttons.
        rf = ttk.Frame(calc_content_frame)# Create a frame for the radio buttons.
        rf.grid(row=0, column=2, sticky="w", padx=2)# Place the frame in the grid.
        ttk.Radiobutton(rf, text="Custom", value="Custom", variable=distance_source_var).pack(side="left")# Create radio button for "Custom" mode.
        ttk.Radiobutton(rf, text="EN Standard", value="EN Standard", variable=distance_source_var).pack(side="left")# Create radio button for "EN Standard" mode.

    # --- Create UI for "Braking distance" mode ---
    elif mode == "distance":# If the selected mode is "distance",
        ttk.Label(calc_content_frame, text="Clamping force (N):").grid(row=0, column=0, sticky="e", padx=2, pady=2)# Label for clamping force input.
        entry_clamping_force = ttk.Entry(calc_content_frame, width=10, validate="key", validatecommand=(vcmd, '%P'))# Create Entry widget for clamping force input.
        entry_clamping_force.grid(row=0, column=1, sticky="w", padx=2, pady=2, ipadx=2, ipady=2)# Place the Entry widget in the grid.
        entry_clamping_force.bind("<Return>", focus_next_widget)# Bind the Enter key to trigger the focus_next_widget function.

        ttk.Label(calc_content_frame, text="Disc dia (mm):").grid(row=1, column=0, sticky="e", padx=2, pady=2)# Label for disc diameter input.
        entry_disc_dia = ttk.Entry(calc_content_frame, width=10, validate="key", validatecommand=(vcmd, '%P'))# Create Entry widget for disc diameter input.
        entry_disc_dia.grid(row=1, column=1, sticky="w", padx=2, pady=2, ipadx=2, ipady=2)# Place the Entry widget in the grid.
        entry_disc_dia.bind("<Return>", focus_next_widget)# Bind the Enter key to trigger the focus_next_widget function.
    
    # --- Default placeholder text ---
    else:# If the selected mode is neither "force" nor "distance",
        ttk.Label(calc_content_frame, text="Coming Soon...").grid(row=0, column=0, sticky="w", padx=2, pady=2)# Display placeholder text.

    # Trigger the function that shows/hides the standard table immediately.
    toggle_standard_table_visibility()# Call the function to show/hide the standard table.

def export_input():# Function to export all current inputs to a CSV file.
    """Gathers all current inputs from the GUI and saves them to a CSV file."""
    pdf = _get_and_validate_inputs()# Get and validate inputs.
    if not pdf:#
        messagebox.showwarning("No Data", "No data to export.")# If no data, display a warning.
        return# If validation fails, exit the function.

    try:# Try to save the data to a CSV file.
        file_path = filedialog.asksaveasfilename(# Get the file path from the user.
            defaultextension=".csv",# Default file extension.
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],# File types for the dialog.
            title="Export Inputs As"# Title of the dialog.
        )# Get the file path from the user.
        if not file_path:# User cancelled the dialog.
            return # User cancelled the dialog.

        # The keys of the pdf dictionary will be the CSV headers.
        headers = pdf.keys()# Get the headers from the dictionary keys.

        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:# Open the file for writing.
            writer = csv.DictWriter(csvfile, fieldnames=headers)# Create a CSV DictWriter object.
            writer.writeheader() # Writes the header row.
            writer.writerow(pdf) # Writes the data row.

        messagebox.showinfo("Success", f"Inputs successfully exported to '{os.path.basename(file_path)}'.")# Show success message.

    except Exception as e:# Handle any exceptions that occur during file operations.
        messagebox.showerror("Export Error", f"Could not save CSV file.\nError: {e}")# Show error message.

def import_vehicle_data():# Function to import Vehicle Data from a selected CSV file.
    """Imports ONLY the Vehicle Data fields from a selected CSV file."""
    try:# Try to import vehicle data from a CSV file.
        file_path = filedialog.askopenfilename(# Get the file path from the user.
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],# File types for the dialog.
            title="Import Vehicle Data from CSV"# Title of the dialog.
        )# Get the file path from the user.
        if not file_path:#  User cancelled the dialog.
            return# User cancelled the dialog.

        with open(file_path, 'r', encoding='utf-8') as csvfile:# Open the CSV file for reading.
            reader = csv.DictReader(csvfile)# Create a CSV DictReader object.
            data_row = next(reader, None) # Get the first row of data.

            if not data_row:# If the file is empty, show a warning.
                messagebox.showwarning("Empty File", "This CSV file is empty.")# Show a warning.
                return# If the file is empty, show a warning.

            # Map the CSV column names to the relevant Entry widgets.
            vehicle_entry_widgets = {
                'mass_kg': entry_mass, 'speed_kmh': entry_speed,
                'number_of_wheels': entry_number_of_wheels,
                'number_braked_wheels': entry_number_braked_wheels,
                'wheel_dia': entry_wheel_dia
            }
            
            # Populate the widgets with data from the file.
            for key, widget in vehicle_entry_widgets.items():# Loop through each Entry widget and populate it if data exists.
                if key in data_row and data_row[key]:# Check if the key exists in the data row and if it has a value.
                    widget.delete(0, tk.END)# Clear the current content of the Entry widget.
                    widget.insert(0, data_row[key])# Insert the data into the Entry widget.
            
    except FileNotFoundError:# Handle file not found error.
        messagebox.showerror("Import Error", "File not found. Please check.")# Show an error message.
    except Exception as e:# Handle any other exceptions.
        messagebox.showerror("Import Error", f"Could not read CSV file.\nError: {e}")# Show an error message.

def export_output_to_csv():# Function to export Calculation Results to a CSV file.
    """Exports all data from the Calculation Results table to a CSV file."""# Step 1: Check if there is data in the table to export.
    
    # Step 1: Check if there is data in the table to export.
    if not tree.get_children():# If the Treeview is empty, show a warning.
        messagebox.showwarning("No Data", "No results to export.")# If the Treeview is empty, show a warning.
        return# User cancelled.

    # Step 2: Ask the user for a file name and location.
    try:# Try to save the data to a CSV file.
        file_path = filedialog.asksaveasfilename(# Get the file path from the user.
            defaultextension=".csv",# Default file extension.
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],# File types for the dialog.
            title="Export Calculation Results As",# Title of the dialog.
            initialfile=f"Braking_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"# Initial file name.
        )# Get the file path from the user.
        if not file_path:#  User cancelled the dialog.
            return # User cancelled.

        # Step 3: Open the file for writing.
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:# Open the CSV file for writing.
            writer = csv.writer(csvfile)# Create a CSV writer object.

            # Step 4: Write the table headers to the file.
            headers = [tree.heading(col)['text'] for col in tree['columns']]# Get the headers from the Treeview column names.
            writer.writerow(headers) # Write the headers row.

            # Step 5: Iterate through the Treeview and write each row to the file.
            for row_id in tree.get_children():# Loop through each row in the Treeview.
                row_values = tree.item(row_id)['values']# Get the values for each column in the current row.
                writer.writerow(row_values)# Write the row values row.
        
        messagebox.showinfo("Success", f"Results successfully exported to '{os.path.basename(file_path)}'.")# Show success message.

    except Exception as e:# Handle any exceptions that occur during file operations.
        messagebox.showerror("Export Error", f"Could not save CSV file.\nError: {e}")# Show error message.

def import_track_data():# Function to import Track Data from a selected CSV file.
    """Imports ONLY the Track Data fields from a selected CSV file."""
    try:
        file_path = filedialog.askopenfilename(# Get the file path from the user.
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],# File types for the dialog.
            title="Import Track Data from CSV"# Title of the dialog.
        )# Get the file path from the user.
        if not file_path:# User cancelled the dialog.
            return# User cancelled the dialog.

        with open(file_path, 'r', encoding='utf-8') as csvfile:# Open the CSV file for reading.
            reader = csv.DictReader(csvfile)# Create a CSV DictReader object.
            data_row = next(reader, None)# Get the first row of data.

            if not data_row:# If the file is empty, show a warning.
                messagebox.showwarning("Empty File", "This CSV file is empty.")# If the file is empty, show a warning.
                return# User cancelled the dialog.

            # Map CSV column names to GUI widgets
            track_widgets = {# Track Data Fields
                # Entry fields
                'gradient_input': entry_gradient,# Gradient Input
                'max_curve': entry_max_curve,# Maximum Curve
                'max_superelevation': entry_superelevation,# Maximum Superelevation
                'max_cant': entry_cant,# Maximum Cant
                'track_gauge': entry_gauge,# Track Gauge
                # Radio button variables
                'gradient_type': gradient_type_var,# Gradient Type Variable
                'max_curve_unit': max_curve_unit_var# Maximum Curve Unit Variable
            }# Map CSV column names to GUI widgets
            
            # Populate the widgets with data
            for key, widget in track_widgets.items():# Loop through each Entry widget and populate it if data exists.
                if key in data_row and data_row[key]:# Check if the key exists in the data row and if it has a value.
                    # If the widget is an Entry
                    if isinstance(widget, ttk.Entry):# If the widget is an Entry
                        widget.delete(0, tk.END)# Clear the current content of the Entry widget.
                        widget.insert(0, data_row[key])# Insert the data into the Entry widget.
                    # If the widget is a StringVar (for radio buttons)
                    elif isinstance(widget, tk.StringVar):# If the widget is a StringVar (for radio buttons)
                        widget.set(data_row[key])#  Set the selected radio button.
            
    except FileNotFoundError:# Handle file not found exception.
        messagebox.showerror("Import Error", "File not found. Please check.")# Show error message.
    except Exception as e:# Handle any other exceptions that occur during file operations.
        messagebox.showerror("Import Error", f"Could not read CSV file.\nError: {e}")# Show error message.

def export_full_report_to_csv():
    """Exports all user inputs and the entire results table to a single CSV file."""
    # Step 1: Get and validate all user inputs.
    pdf = _get_and_validate_inputs()# Get and validate all user inputs.
    if not pdf:# If there is no input data, show a warning and return.
        messagebox.showwarning("No Input Data", "No input data for report.")
        return

    # Step 2: Check if there is data in the results table.
    if not tree.get_children():# get_children() returns an empty list if the Treeview is empty.
        messagebox.showwarning("No Output Data", "No results for report. Please calculate first.")# If there is no output data, show a warning and return.
        return

    # Step 3: Ask the user for a file name and location.
    try:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",# Default file extension.
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],# File types for the dialog.
            title="Export Full Report As", # Title of the dialog.
            initialfile=f"Full_Braking_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"# Initial file name.
        )# Get the file path from the user.
        if not file_path:
            return # User cancelled.

        # Step 4: Open the file for writing.
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:# Open the CSV file for writing.
            writer = csv.writer(csvfile)# Create a CSV writer object.

            # --- Write the Inputs Section ---
            writer.writerow(['--- CALCULATION INPUTS ---'])# Write section header.
            writer.writerow(['Parameter', 'Value'])# Write table headers.
            for key, value in pdf.items():# Loop through each input parameter and its value.
                writer.writerow([key, value if value is not None else ""])# Write each parameter and its value.
            
            # Add a separator for clarity
            writer.writerow([])# Add a separator.
            writer.writerow([])# Add a separator.

            # --- Write the Results Table Section ---
            writer.writerow(['--- CALCULATION RESULTS ---'])# Write section header.
            # Write the table headers
            headers = [tree.heading(col)['text'] for col in tree['columns']]
            writer.writerow(headers)# Write table headers.

            # Write each row from the Treeview
            for row_id in tree.get_children():# Loop through each row in the Treeview.
                row_values = tree.item(row_id)['values']# Get the values for each column in the row.
                writer.writerow(row_values)# Write each row's values.
        
        messagebox.showinfo("Success", f"Full report successfully exported to '{os.path.basename(file_path)}'.")# Show success message.

    except Exception as e:# Handle any exceptions that occur during file operations.
        messagebox.showerror("Export Error", f"Could not save CSV file.\nError: {e}")# Show error message.

def toggle_standard_table_visibility(*args):
    """
    Shows or hides the standard table display and the compliance column based on the radio button state.
    Also enables/disables the Custom Distance input field accordingly.
    """
    selected_option = distance_source_var.get()# Get the selected radio button option.

    # Define columns to show when custom is selected (all except the compliance column)
    custom_columns = (
        'speed_kmh', 'speed_ms', 'reaction_dist', 'app_braking_force', 
        'grav_force', 'net_force', 'deceleration', 'braking_dist', 'total_stop_dist'
    )

    if selected_option == "EN Standard":
        # 1. Show the standard table frame.
        results_image_frame.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        # 2. Make the main results table smaller.
        tree_frame.grid_configure(columnspan=1)
        # 3. Adjust the column weights of the container frame to a 4:1 ratio.
        bottom_tables_container.columnconfigure(0, weight=4)
        bottom_tables_container.columnconfigure(1, weight=1)
        # 4. Disable the custom distance entry field.
        if entry_distance_custom:
            entry_distance_custom.config(state="disabled")
        # 5. Show all columns, including the compliance one.
        tree.config(displaycolumns='#all')
        # 6. Change column header from "Applied Force" to "Max Braking Force"
        tree.heading('app_braking_force', text='Max Braking Force (N)')


    else: # This runs when "Custom" is selected.
        # 1. Hide the standard table frame.
        results_image_frame.grid_forget()
        # 2. Make the main results table take up the full width.
        tree_frame.grid_configure(columnspan=2)
        # 3. Give all the container weight to the first column.
        bottom_tables_container.columnconfigure(0, weight=1)
        bottom_tables_container.columnconfigure(1, weight=0)
        # 4. Enable the custom distance entry field.
        if entry_distance_custom:
            entry_distance_custom.config(state="normal")
        # 5. Hide the compliance column by showing only the others.
        tree.config(displaycolumns=custom_columns)
        # 6. Change column header from "Max Braking Force" to "Applied Force"
        tree.heading('app_braking_force', text='Required Force (N)')


def create_and_save_pdf_report():# 
    # Check if the "Calculate" button has been run first and data was captured
    if not pdf_report_placeholders:
        messagebox.showwarning("No Data", "No data for report. Please calculate first.")# If there is no data, show a warning and
        return

    try:# Try to generate and save the PDF report.
        file_path = filedialog.asksaveasfilename(# Get the file path from the user.
            defaultextension=".pdf",# Default file extension.
            filetypes=[("PDF Documents", "*.pdf"), ("All Files", "*.*")],# File types for the dialog.
            title="Save PDF Report As",# Title of the dialog.
            initialfile=f"Braking_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )# Get the file path from the user.
        if not file_path:# User cancelled the dialog.
            return

        # Pass the already-captured dictionary directly to the template generator
        success = _generate_pdf_from_template(file_path, pdf_report_placeholders)# Pass the already-captured dictionary directly to the template generator.

        if success:# If PDF generation was successful, inform the user.
            messagebox.showinfo("Success", f"PDF report successfully generated:\n'{os.path.basename(file_path)}'")# Show success message.
            if messagebox.askyesno("Open Report", "Do you want to open the generated PDF report?"):# Ask the user if they want to open the generated PDF report.
                os.startfile(file_path)# Open the generated PDF report.

    except Exception as e:# Handle any exceptions that occur during PDF generation.
        messagebox.showerror("Report Generation Error", f"Could not make PDF.\nError: {e}")# Show error message.


def _generate_pdf_from_template(filename, results):# Function to generate a PDF report from a LaTeX template.
    try:# Try to generate the PDF report.
        env = Environment(loader=FileSystemLoader('.'))# Set up Jinja2 environment to load templates from the current directory.
        template = env.get_template('template.tex')# Load the LaTeX template file.
        
        # Make a copy of the results to avoid changing the original
        context = results.copy()
        
        # Get the author names from the GUI, escape them, and add them to the context
        context.update({
            'made_by': escape_latex(entry_made_by.get()),
            'checked_by': escape_latex(entry_checked_by.get()),
            'approved_by': escape_latex(entry_approved_by.get())
        })

        rendered_tex = template.render(context)# Render the template with the context.
        temp_tex_filename = "temp_report.tex"# Name of the temporary LaTeX file.
        with open(temp_tex_filename, 'w', encoding='utf-8') as f:# Write the rendered LaTeX to the temporary file.
            f.write(rendered_tex)# Write the rendered LaTeX to the temporary file.
        
        # Run pdflatex command
        process = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', '-output-directory', '.', temp_tex_filename],
            capture_output=True, text=True, encoding='utf-8', errors='ignore'
        )# Run pdflatex command.
        
        if process.returncode != 0:# If pdflatex returned a non-zero exit code, show an error message and return False.
            print(f"LaTeX compilation error: {process.stdout}")# Print LaTeX compilation error and return False.
            messagebox.showerror("PDF Generation Error", "Could not create PDF. Check LaTeX logs for details.")# Show error message and return False.
            return False# Return False.

        # Clean up temporary files and move PDF to final location
        generated_pdf = temp_tex_filename.replace('.tex', '.pdf')# Name of the generated PDF file.
        if os.path.exists(filename):# If the target file already exists, remove it.
            os.remove(filename)# Remove the existing target file.
        
        # Use shutil.move() instead of os.rename() to support cross-drive moves
        shutil.move(generated_pdf, filename)# Move the generated PDF to the desired location (works across drives).
        
        for ext in ['.aux', '.log', '.tex']:# Clean up temporary files.
            temp_file = temp_tex_filename.replace('.tex', ext)# Get temporary file name.
            if os.path.exists(temp_file):# If the temporary file exists, remove it.
                os.remove(temp_file)# Remove the temporary file.
                
        return True# Return True.
    except FileNotFoundError:# Handle the case where pdflatex is not found.
        messagebox.showerror(
            "Dependency Error",
            "LaTeX not found. Please install a LaTeX distribution (for example, MiKTeX on Windows or TeX Live on Linux) "
            "and make sure the 'pdflatex' executable is available on your system PATH so the application can generate PDF reports."
        )
        return False# Return False.
    except Exception as e:# Handle any other exceptions that occur during PDF generation.
        messagebox.showerror("PDF Generation Error", f"Could not make PDF.\nError: {e}")# Show error message and return False.
        return False# Return False.

def escape_latex(s):# Function to escape special LaTeX characters in a string to prevent compilation errors.
    """Escapes special LaTeX characters in a string to prevent compilation errors."""
    if not isinstance(s, str):#  If the input is not a string, return it as is.
        return s
    mapping = {# Convert special LaTeX characters to their LaTeX escape sequences.
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
        '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}', '\\': r'\textbackslash{}',';': r'\;',':': r'\:',
    }
    return re.sub(r'[&%$#_{}~^;:\\\\]', lambda m: mapping.get(m.group(0)), s)# Return the escaped string.

# =============================================================================
# GUI SETUP
# =============================================================================
# --- Main Window Creation ---
root = tk.Tk()# Create the main application window.
root.title("Braking Performance Calculator")# Set the window title.
root.geometry("1560x785")# Set the initial window size.
root.state('zoomed') # Start the application maximized.
root.minsize(1500, 500)# Set a minimum window size.
root.configure(bg="#f0f0f0") # Set a light grey background color.
try:# Try to set a custom window icon.
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    icon_path = os.path.join(script_dir, "logo.png.ico")
    root.iconbitmap(icon_path)
except Exception:
    # Silently fail if icon is not found
    pass

# --- Style Configuration ---
style = ttk.Style()# Create a style object for theming.
style.theme_use("clam") # Use a modern theme.

# --- General Widget Styles ---
style.configure("TLabel", background="#f0f0f0", foreground="#333333", font=("Helvetica", 10))# General label style.
style.configure("TFrame", background="#f0f0f0")# General frame style.
style.configure("TLabelframe", background="#f0f0f0", bordercolor="#cccccc")# General labelframe style.
style.configure("TLabelframe.Label", background="#f0f0f0", foreground="#00529B", font=("Helvetica", 12, "bold"))# Labelframe label style.
style.configure("TEntry", font=("Helvetica", 10))# Entry style.
style.map("TEntry", fieldbackground=[('focus', '#eaf4fe')]) # Light blue focus color.
style.configure("TRadiobutton", background="#f0f0f0", foreground="#333333", font=("Helvetica", 10)) # <<< CHANGED: Radiobutton background

# --- Button Styles ---
style.configure("TButton", foreground="white", font=("Helvetica", 10, "bold"), padding=6)# General button style.
style.configure("Green.TButton", background="#28a745", bordercolor="#28a745") # Green for 'Calculate'
style.map("Green.TButton", background=[('active', '#218838')])# Green for 'Calculate'
style.configure("Blue.TButton", background="#007bff", bordercolor="#007bff") # Blue for 'Report'
style.map("Blue.TButton", background=[('active', '#0069d9')])# Blue for 'Report'
style.configure("Orange.TButton", background="#fd7e14", foreground="white", bordercolor="#fd7e14") # Orange for 'Export'
style.map("Orange.TButton", background=[('active', '#e86100')])# Orange for 'Export'


# --- Treeview (Results Table) Styles ---
style.configure("Treeview.Heading", font=("Helvetica", 10, "bold"), background="#4F4F4F", foreground="white", relief="flat")# 
style.map('Treeview.Heading', background=[('active', '#6a6a6a')])# Heading background color.
style.configure("Treeview", background="white", foreground="#333333", fieldbackground="white", rowheight=28, font=("Helvetica", 10))# Table style.
style.map('Treeview', background=[('selected', '#BEE6FF')], foreground=[('selected', 'black')])# Selected row style.

# --- Input Validation Command ---
vcmd = root.register(validate_input)

# --- Main Layout Frames ---
# The main_frame holds all other widgets.
main_frame = ttk.Frame(root, padding=2)# The main_frame holds all other widgets.
main_frame.pack(fill="both", expand=True)# Pack to fill the entire window.

# Configure the main frame's grid: 1 column, 3 rows.
main_frame.columnconfigure(0, weight=1) # Column 0 will expand horizontally.
main_frame.rowconfigure(0, weight=0)  # Title row has a fixed height.
main_frame.rowconfigure(1, weight=0)  # Input row has a fixed height.
main_frame.rowconfigure(2, weight=1)  # Tables row will expand vertically to fill space.

# --- Top Frame (for the application title and authors) ---
top_frame = ttk.Frame(main_frame)#  The top frame holds the title and author inputs.
top_frame.grid(row=0, column=0, sticky="ew", padx=2, pady=2)# Pack to fill the top frame.
top_frame.columnconfigure(0, weight=1) # Let the title/author column take up space
top_frame.columnconfigure(1, weight=0) # Column for the button

ttk.Label(top_frame, text="Braking Performance Calculator", font=("Helvetica", 16, "bold")).grid(row=0, column=0, sticky="w", padx=2, pady=2)# Application title.

#  Author Inputs Frame ---
author_frame = ttk.Frame(top_frame)# The author inputs frame holds the author input fields.
author_frame.grid(row=1, column=0, sticky="ew", pady=2, padx=2)#    Pack to fill the author frame.

#  THESE FOUR LINES ---
ttk.Label(author_frame, text="Doc-No.:").pack(side="left", padx=2, pady=2)# Doc-No. label.
entry_doc_no = ttk.Entry(author_frame, width=20)# Doc-No. entry.
entry_doc_no.pack(side="left", padx=2, pady=2, ipadx=2, ipady=2)
entry_doc_no.bind("<Return>", focus_next_widget)# Bind return key to next widget.

ttk.Label(author_frame, text="Made by:").pack(side="left", padx=2, pady=2)# Made by label.
entry_made_by = ttk.Entry(author_frame, width=20)# Made by entry.
entry_made_by.pack(side="left", padx=2, pady=2, ipadx=2, ipady=2)
entry_made_by.bind("<Return>", focus_next_widget)# Bind return key to next widget.

ttk.Label(author_frame, text="Checked by:").pack(side="left", padx=2, pady=2)# checked by label.
entry_checked_by = ttk.Entry(author_frame, width=20)# Checked by entry.
entry_checked_by.pack(side="left", padx=2, pady=2, ipadx=2, ipady=2)
entry_checked_by.bind("<Return>", focus_next_widget)# Bind return key to next widget.

ttk.Label(author_frame, text="Approved by:").pack(side="left", padx=2, pady=2)# Approved by label.
entry_approved_by = ttk.Entry(author_frame, width=20)# Approved by entry.
entry_approved_by.pack(side="left", padx=2, pady=2, ipadx=2, ipady=2)
entry_approved_by.bind("<Return>", focus_next_widget)# bind return key to next widget.


# --- PDF Report Button ---
pdf_report_button = ttk.Button(top_frame, text="Generate PDF Report", command=create_and_save_pdf_report, style="Blue.TButton")# button to generate a PDF report. 
pdf_report_button.grid(row=0, column=1, rowspan=2, sticky='nswe', padx=2, pady=2, ipadx=2, ipady=2)
# --- Top Inputs Container (holds all input frames) ---
top_inputs_container = ttk.Frame(main_frame)# Holds all input frames.
top_inputs_container.grid(row=1, column=0, sticky="ew", pady=2, padx=2)
top_inputs_container.columnconfigure((0, 1, 2, 3), weight=1) # Four equal-width columns.

# --- Vehicle Data Frame (Column 0) ---
vehicle_frame = ttk.LabelFrame(top_inputs_container, text=" 1. Vehicle Data ", padding=2)
vehicle_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=2, pady=2)
vehicle_frame.columnconfigure(1, weight=1)
# Button to import vehicle data from a CSV file.
import_button = ttk.Button(vehicle_frame, text="Import Vehicle Data (CSV)", command=import_vehicle_data, style="Orange.TButton") 
import_button.grid(row=0, column=0, columnspan=2, sticky='ew', padx=2, pady=2, ipadx=2, ipady=2)
# Labels and Entry widgets for vehicle parameters.
ttk.Label(vehicle_frame, text="GVW (kg):").grid(row=1, column=0, padx=2, pady=2, sticky="w")
entry_mass = ttk.Entry(vehicle_frame, width=10, validate="key", validatecommand=(vcmd, '%P'))
entry_mass.grid(row=1, column=1, padx=2, pady=2, sticky="w", ipadx=2, ipady=2)
ttk.Label(vehicle_frame, text="Max Speed (km/h):").grid(row=2, column=0, padx=2, pady=2, sticky="w")
entry_speed = ttk.Entry(vehicle_frame, width=10, validate="key", validatecommand=(vcmd, '%P'))
entry_speed.grid(row=2, column=1, padx=2, pady=2, sticky="w", ipadx=2, ipady=2)
ttk.Label(vehicle_frame, text="Driving Wheels:").grid(row=3, column=0, padx=2, pady=2, sticky="w")
entry_number_of_wheels = ttk.Entry(vehicle_frame, width=10, validate="key", validatecommand=(vcmd, '%P'))
entry_number_of_wheels.grid(row=3, column=1, padx=2, pady=2, sticky="w", ipadx=2, ipady=2)
ttk.Label(vehicle_frame, text="Braked Wheels:").grid(row=4, column=0, padx=2, pady=2, sticky="w")
entry_number_braked_wheels = ttk.Entry(vehicle_frame, width=10, validate="key", validatecommand=(vcmd, '%P'))
entry_number_braked_wheels.grid(row=4, column=1, padx=2, pady=2, sticky="w", ipadx=2, ipady=2)
ttk.Label(vehicle_frame, text="Wheel Dia (mm):").grid(row=5, column=0, padx=2, pady=2, sticky="w")
entry_wheel_dia = ttk.Entry(vehicle_frame, width=10, validate="key", validatecommand=(vcmd, '%P'))
entry_wheel_dia.grid(row=5, column=1, padx=2, pady=2, sticky="w", ipadx=2, ipady=2)

# --- Track Data Frame (Column 1) ---
track_frame = ttk.LabelFrame(top_inputs_container, text=" 2. Track Data ", padding=2)
track_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=2, pady=2)
track_frame.columnconfigure(0, weight=1)

# Button to import Track data from a CSV file.
import_track_button = ttk.Button(track_frame, text="Import Track Data (CSV)", command=import_track_data, style="Orange.TButton" )
import_track_button.grid(row=0, column=0, sticky='ew', padx=2, pady=2, ipadx=2, ipady=2)

# Radio buttons to select track standard (CUSTOM, IR, HSR).
ttk.Label(track_frame, text="Choose Track:").grid(row=1, column=0, padx=2, pady=2, sticky="w")
track_radio_frame = ttk.Frame(track_frame)
track_radio_frame.grid(row=2, column=0, sticky="ew", padx=2, pady=2)
track_standard_var = tk.StringVar(value="CUSTOM")
ttk.Radiobutton(track_radio_frame, text="CUSTOM", value="CUSTOM", variable=track_standard_var).pack(side="left", padx=2, ipadx=2, ipady=2)
ttk.Radiobutton(track_radio_frame, text="IR", value="IR", variable=track_standard_var).pack(side="left", padx=2, ipadx=2, ipady=2)
ttk.Radiobutton(track_radio_frame, text="HSR", value="HSR", variable=track_standard_var).pack(side="left", padx=2, ipadx=2, ipady=2)
track_standard_var.trace_add("write", update_track_data_ui)

# --- Container for the detailed track input fields ---
track_inputs_container = ttk.Frame(track_frame)
track_inputs_container.grid(row=3, column=0, sticky="ew", pady=2, padx=2)
track_inputs_container.columnconfigure(2, weight=1)
track_inputs_container.columnconfigure(3, weight=1)

# --- ROW 0: Max Gradient ---
ttk.Label(track_inputs_container, text="Max Gradient Value:").grid(row=0, column=0, padx=2, pady=2, sticky="w")
entry_gradient = ttk.Entry(track_inputs_container, width=10, validate="key", validatecommand=(vcmd, '%P'))
entry_gradient.grid(row=0, column=1, padx=2, pady=2, sticky="w", ipadx=2, ipady=2)
gradient_type_var = tk.StringVar(value="1 in G")
ttk.Label(track_inputs_container, text="Unit:").grid(row=0, column=2, padx=2, pady=2, sticky="w")
grad_types_frame = ttk.Frame(track_inputs_container)
grad_types_frame.grid(row=0, column=3, padx=2, pady=2, sticky="w")
ttk.Radiobutton(grad_types_frame, text="Degree(°)", value="Degree (°)", variable=gradient_type_var).pack(side="left")
ttk.Radiobutton(grad_types_frame, text="1 in G", value="1 in G", variable=gradient_type_var).pack(side="left")
ttk.Radiobutton(grad_types_frame, text="Percentage(%)", value="Percentage (%)", variable=gradient_type_var).pack(side="left")

# --- ROW 1: Max Curve ---
ttk.Label(track_inputs_container, text="Max Curve:").grid(row=1, column=0, padx=2, pady=2, sticky="w")
entry_max_curve = ttk.Entry(track_inputs_container, width=10, validate="key", validatecommand=(vcmd, '%P'))
entry_max_curve.grid(row=1, column=1, padx=2, pady=2, sticky="w", ipadx=2, ipady=2)
ttk.Label(track_inputs_container, text="Unit:").grid(row=1, column=2, padx=2, pady=2, sticky="w")
max_curve_unit_var = tk.StringVar(value="m")
curve_unit_frame = ttk.Frame(track_inputs_container)
curve_unit_frame.grid(row=1, column=3, padx=2, pady=2, sticky="w")
ttk.Radiobutton(curve_unit_frame, text="Meter", value="m", variable=max_curve_unit_var).pack(side="left")
ttk.Radiobutton(curve_unit_frame, text="Degree(°)", value="degree", variable=max_curve_unit_var).pack(side="left")

# --- ROW 2: Friction (µ) and Road Mode Checkbox ---
ttk.Label(track_inputs_container, text="Friction (µ):").grid(row=2, column=0, padx=2, pady=2, sticky="w")
friction_var = tk.StringVar(value="0.7")
entry_friction = ttk.Entry(track_inputs_container, width=10, textvariable=friction_var, validate="key", validatecommand=(vcmd, '%P'))
entry_friction.grid(row=2, column=1, padx=2, pady=2, sticky="w", ipadx=2, ipady=2)

# Checkbox to enable/disable Road Mode calculations
road_mode_enabled_var = tk.BooleanVar(value=True)
road_mode_checkbox = ttk.Checkbutton(track_inputs_container, text="Calculate Road Mode (Friction Based)", variable=road_mode_enabled_var)
road_mode_checkbox.grid(row=2, column=2, columnspan=2, padx=2, pady=2, sticky="w")

# --- ROW 3: Max Superelevation & Image Frame Start ---
ttk.Label(track_inputs_container, text="Max Superelevation (mm):").grid(row=3, column=0, padx=2, pady=2, sticky="w")
entry_superelevation = ttk.Entry(track_inputs_container, width=10, validate="key", validatecommand=(vcmd, '%P'))
entry_superelevation.grid(row=3, column=1, padx=2, pady=2, sticky="w", ipadx=2, ipady=2)

# The image frame is now created and placed logically with the content of row 3
track_image_frame = ttk.Frame(track_inputs_container)
track_image_frame.grid(row=3, column=2, rowspan=3, columnspan=2, sticky="nsew", padx=2, pady=2)
track_image_frame.columnconfigure(0, weight=1)
track_image_frame.rowconfigure(0, weight=1)
track_image_label = ttk.Label(track_image_frame, background="#ffffff", relief="solid", borderwidth=1, text="Image Here")
track_image_label.grid(row=0, column=0, sticky="nsew")
track_image_label.bind("<Double-Button-1>", on_image_double_click)
track_label = ttk.Label(track_image_frame, text="Cant", font=("Helvetica", 10, "bold"))
track_label.grid(row=1, column=0, sticky="n")

# --- ROW 4: Max Cant ---
ttk.Label(track_inputs_container, text="Max Cant (mm):").grid(row=4, column=0, padx=2, pady=2, sticky="w")
entry_cant = ttk.Entry(track_inputs_container, width=10, validate="key", validatecommand=(vcmd, '%P'))
entry_cant.grid(row=4, column=1, padx=2, pady=2, sticky="w", ipadx=2, ipady=2)

# --- ROW 5: Track Gauge ---
ttk.Label(track_inputs_container, text="Track Gauge (mm):").grid(row=5, column=0, padx=2, pady=2, sticky="w")
entry_gauge = ttk.Entry(track_inputs_container, width=10, validate="key", validatecommand=(vcmd, '%P'))
entry_gauge.grid(row=5, column=1, padx=2, pady=2, sticky="w", ipadx=2, ipady=2)
# --- Brake Type & Calculation Mode Frame (Column 2) ---
brake_type_frame = ttk.LabelFrame(top_inputs_container, text=" 3. Choose brake type ", padding=2)
brake_type_frame.grid(row=0, column=2, sticky="nsew", padx=2, pady=2)
brake_type_frame.columnconfigure(3, weight=1)
brake_type_var = tk.StringVar(value="Disc Brake")
ttk.Label(brake_type_frame, text="Type:").grid(row=0, column=0, padx=2, pady=2, sticky="w")
ttk.Radiobutton(brake_type_frame, text="Disc Brake", value="Disc Brake", variable=brake_type_var).grid(row=0, column=1, padx=2, pady=2, sticky="w")
ttk.Radiobutton(brake_type_frame, text="Tread Brake", value="Tread Brake", variable=brake_type_var).grid(row=0, column=2, padx=2, pady=2, sticky="w")
brake_image_small_label = ttk.Label(brake_type_frame, background="#ffffff")
brake_image_small_label.grid(row=0, column=3, padx=2, pady=2, sticky="e")
brake_type_var.trace_add("write", update_brake_image)

calc_frame = ttk.LabelFrame(top_inputs_container, text=" 4. Calculate ", padding=2)
calc_frame.grid(row=1, column=2, sticky="nsew", padx=2, pady=2)
calc_frame.columnconfigure(1, weight=1)
calc_mode_var = tk.StringVar(value="force")
# Radio buttons to switch between different calculation modes.
ttk.Radiobutton(calc_frame, text="Braking force", value="force", variable=calc_mode_var).grid(row=0, column=0, sticky="w", padx=2, pady=2)
ttk.Radiobutton(calc_frame, text="Braking distance", value="distance", variable=calc_mode_var).grid(row=1, column=0, sticky="w", padx=2, pady=2)
ttk.Radiobutton(calc_frame, text="Brake details", value="details", variable=calc_mode_var).grid(row=2, column=0, sticky="w", padx=2, pady=2)
# This frame will be dynamically populated by `update_calc_mode_ui`.
calc_content_frame = ttk.Frame(calc_frame)
calc_content_frame.grid(row=0, column=1, rowspan=3, sticky="nsew", padx=2, pady=2)
calc_content_frame.columnconfigure(1, weight=1)

# Variables to control dynamic UI elements.
distance_source_var = tk.StringVar(value="EN Standard")
distance_source_var.trace_add("write", toggle_standard_table_visibility)
entry_distance_custom, entry_clamping_force, entry_disc_dia = None, None, None
calc_mode_var.trace_add("write", update_calc_mode_ui)


# --- Actions Frame (Column 3) ---
validate_speed_cmd = root.register(validate_speed_increment)
validate_gradient_cmd = root.register(validate_gradient_increment)
actions_frame = ttk.LabelFrame(top_inputs_container, text=" 5. Actions ", padding=2)
actions_frame.grid(row=0, column=3, rowspan=2, sticky="nsew", padx=2, pady=2)
actions_frame.columnconfigure(1, weight=1)
# Widgets for setting the number of increments for loops.
ttk.Label(actions_frame, text="Analysis Steps", font=("Helvetica", 10, "bold")).grid(row=0, column=0, columnspan=2, pady=2, padx=2)
ttk.Label(actions_frame, text="Speed Increment (km/h):").grid(row=1, column=0, padx=2, pady=2, sticky="w")
entry_speed_increment = ttk.Entry(actions_frame, width=6, validate="key", validatecommand=(validate_speed_cmd, '%P'))
entry_speed_increment.grid(row=1, column=1, padx=2, pady=2, sticky="ew", ipadx=2, ipady=2)
ttk.Label(actions_frame, text="Number of Division Gradient:").grid(row=2, column=0, padx=2, pady=2, sticky="w")
entry_gradient_increment = ttk.Entry(actions_frame, width=6, validate="key", validatecommand=(validate_gradient_cmd, '%P'))
entry_gradient_increment.grid(row=2, column=1, padx=2, pady=2, sticky="ew", ipadx=2, ipady=2)
ttk.Separator(actions_frame, orient='horizontal').grid(row=3, column=0, columnspan=2, sticky='ew', pady=2, padx=2)
# Main action buttons.
calculate_button = ttk.Button(actions_frame, text="Calculate", command=convert, style="Green.TButton")
calculate_button.grid(row=4, column=0, columnspan=2, sticky='ew', padx=2, pady=2, ipadx=2, ipady=2)
export_button = ttk.Button(actions_frame, text="Export Input (CSV)", command=export_input, style="Blue.TButton")
export_button.grid(row=5, column=0, columnspan=2, sticky='ew', padx=2, pady=2, ipadx=2, ipady=2)
export_output_button = ttk.Button(actions_frame, text="Export Output (CSV)", command=export_output_to_csv, style="Blue.TButton")
export_output_button.grid(row=6, column=0, columnspan=2, sticky='ew', padx=2, pady=2, ipadx=2, ipady=2)
# --- NEW: Full Report Button ---
full_report_button = ttk.Button(actions_frame, text="Export Full Report (CSV)", command=export_full_report_to_csv, style="Blue.TButton")
full_report_button.grid(row=7, column=0, columnspan=2, sticky='ew', padx=2, pady=2, ipadx=2, ipady=2)


# --- Bottom Tables Container (holds the two main tables) ---
bottom_tables_container = ttk.Frame(main_frame)
bottom_tables_container.grid(row=2, column=0, sticky="nsew", padx=2, pady=2)
bottom_tables_container.columnconfigure(0, weight=4) # Main table gets 4x the width.
bottom_tables_container.columnconfigure(1, weight=1) # Standard table gets 1x the width.
bottom_tables_container.rowconfigure(0, weight=1)

# --- Calculation Results Table (Treeview) ---
tree_frame = ttk.LabelFrame(bottom_tables_container, text=" Calculation Results ", padding=2)
tree_frame.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
tree_frame.rowconfigure(0, weight=1)
tree_frame.columnconfigure(0, weight=1)

columns = (
    'speed_kmh', 'speed_ms', 'reaction_dist', 'app_braking_force', 
    'grav_force', 'net_force', 'deceleration', 'braking_dist', 'total_stop_dist', 
    'std_compliance'
)
tree = ttk.Treeview(tree_frame, columns=columns, show="headings")

# Set the text and properties for each column heading.
tree.heading('speed_kmh', text='Speed (km/h)')
tree.heading('speed_ms', text='Speed (m/s)')
tree.heading('reaction_dist', text='Reaction Dist (m)')
tree.heading('app_braking_force', text='Applied Force (N)')
tree.heading('grav_force', text='Gravitational Force (N)')
tree.heading('net_force', text='Net Force (N)')
tree.heading('deceleration', text='Deceleration (m/s²)')
tree.heading('braking_dist', text='Braking Dist (m)')
tree.heading('total_stop_dist', text='Total Stop Dist (m)')
tree.heading('std_compliance', text='Standard Compliance')


# Set the width and alignment for each column.
tree.column('speed_kmh', anchor="center", width=150)
tree.column('speed_ms', anchor="center", width=120)
tree.column('reaction_dist', anchor="center", width=130)
tree.column('app_braking_force', anchor="center", width=140)
tree.column('grav_force', anchor="center", width=170)
tree.column('net_force', anchor="center", width=120)
tree.column('deceleration', anchor="center", width=130)
tree.column('braking_dist', anchor="center", width=120)
tree.column('total_stop_dist', anchor="center", width=140)
tree.column('std_compliance', anchor="center", width=150)


# Add horizontal and vertical scrollbars to the table.
h_scrollbar_tree = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
tree.configure(xscrollcommand=h_scrollbar_tree.set)
scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
tree.configure(yscrollcommand=scrollbar.set)

# Place the table and scrollbars in the grid.
tree.grid(row=0, column=0, sticky="nsew")
scrollbar.grid(row=0, column=1, sticky="nsew")
h_scrollbar_tree.grid(row=1, column=0, sticky="ew")

#  styles for the group headers
# Configure tags for alternating row colors.
tree.tag_configure("oddrow", background="#f9f9f9") # Very light grey
tree.tag_configure("evenrow", background="#ffffff") # White

# Style for the "Gradient" row
tree.tag_configure("gradient_header", background="#343a40",foreground="white", font=('Helvetica', 11, 'bold'))
# Style for the "Moving Up / Moving Down" row
tree.tag_configure("direction_header", background="#007bff", foreground="white", font=('Helvetica', 10, 'italic'))

# --- Standard Stopping Distance Table (Text Widget) ---
results_image_frame = ttk.LabelFrame(bottom_tables_container, text=" Standard Stopping Distance Table ", padding=2)
results_image_frame.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
results_image_frame.rowconfigure(0, weight=1)
results_image_frame.columnconfigure(0, weight=1) # Text widget ko expand hone dein

standard_text_display = tk.Text(results_image_frame, height=20, wrap="word", relief="sunken", borderwidth=1, width=45)
standard_text_display.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

standard_scrollbar = ttk.Scrollbar(results_image_frame, orient="vertical", command=standard_text_display.yview)
standard_scrollbar.grid(row=0, column=1, sticky="ns", padx=2, pady=2)

standard_text_display.configure(yscrollcommand=standard_scrollbar.set)


# =============================================================================
# INITIALIZATION & MAIN LOOP
# =============================================================================
# --- Create a Status Bar for Widget Inspection ---
status_bar_var = tk.StringVar()
status_bar = ttk.Label(
    root, 
    textvariable=status_bar_var, 
    relief=tk.SUNKEN, 
    anchor="w", 
    padding=2
)
status_bar.pack(side="bottom", fill="x")

# --- Bindings for Keyboard Navigation ---
# Create a list of all static Entry widgets for easy binding.
all_entries = [
    entry_mass, entry_speed, entry_number_of_wheels, entry_number_braked_wheels,
    entry_wheel_dia, entry_gradient, entry_max_curve, entry_superelevation,
    entry_cant, entry_gauge, entry_speed_increment, entry_gradient_increment
]
# Bind the <Return> (Enter) key to move focus to the next widget for all entries.
for entry in all_entries:
    entry.bind("<Return>", focus_next_widget)

# Bind Enter key on the Calculate button to trigger the calculation.
calculate_button.bind("<Return>", lambda e: calculate_button.invoke())
# Bind Control+Enter globally to trigger the calculation from anywhere.
root.bind('<Control-Return>', lambda e: calculate_button.invoke())

# --- NEW: Bind focus events to track data inputs to update the image ---
entry_gradient.bind("<FocusIn>", update_track_image)
entry_max_curve.bind("<FocusIn>", update_track_image)
entry_superelevation.bind("<FocusIn>", update_track_image)
entry_cant.bind("<FocusIn>", update_track_image)
entry_gauge.bind("<FocusIn>", update_track_image)
# --- Bind hover events to all widgets for inspection ---
# First, create a list of all widgets in the application.
all_widgets = []
# Start with the direct children of the main window.
widgets_to_check = list(root.winfo_children())

# # This while loop will continue as long as there are widgets to check.
# while widgets_to_check:
#     # Get the next widget from the list.
#     widget = widgets_to_check.pop(0)
#     # Add it to our complete list.
#     all_widgets.append(widget)
#     # Add its direct children to the list of widgets we need to check.
#     widgets_to_check.extend(widget.winfo_children())

# # Now, loop through the complete list of all widgets and bind the events.
# for widget in all_widgets:
#     widget.bind("<Enter>", on_widget_enter)
#     widget.bind("<Leave>", on_widget_leave)

# --- Initialize the UI State on Startup ---
update_calc_mode_ui()           # Set up the initial dynamic 'Calculate' section.
populate_standard_table_display() # Fill the standard table text widget.
update_brake_image()            # Show the initial brake image.
update_track_data_ui()          # Set the initial state of the track data fields.
toggle_standard_table_visibility() # Set initial table visibility and column headers.
entry_mass.focus_set()          # Set the initial focus to the first input field.
update_track_image() # Display the default image on startup
Tooltip(track_image_label, "Double tap to zoom in")
on_image_double_click.zoom_window = None # Initialize a global variable to keep track of the zoom window

# --- Run the Application ---
# root.mainloop() starts the Tkinter event loop, making the GUI interactive.
root.mainloop()

