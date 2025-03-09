import os
import sys
import argparse
import glob
import json
import datetime
import io  
import base64 
import numpy as np 
import math 
import shutil 
import pandas as pd  
from scipy.stats import zscore  
from matplotlib.backends.backend_pdf import FigureCanvasPdf as FigureCanvas 

def show_help():
    help_message = """
    Usage: python fastMRIQC.py -p <SubjectID> [-b <BIDS_ROOT>] [-s <SESSION>] [-o <QC_ROOT>] [-h]

    Arguments:
      -p <SubjectID>  : Mandatory. Specify the subject ID to be processed.
      -b <BIDS_ROOT>  : Optional. Specify the root directory of the BIDS dataset. Defaults to the current working directory.
      -s <SESSION>    : Optional. Specify a particular session to process. If not provided, all sessions for the subject will be processed.
      -o <QC_ROOT>    : Optional. Specify the output directory for QC reports. Defaults to a 'qc' directory adjacent to the BIDS root.
      -h, --help      : Display this help message and exit.


    # ------------------------------------------------------------------------

    This script automates few basic quality control checks of MRI data.
    It reviews the content in anatomical (anat), diffusion-weighted imaging (dwi), and functional (func) folders of a BIDS formatted MRI dataset,
    generating a QC report in HTML format (typically all under 2-3 minutes).

    The goal is to complete this quick version of QC on the same day as MRI acquisition, allowing for timely review and decision-making.

    # ------------------------------------------------------------------------

    Requirements:
      - BIDS-formatted data: The MRI data must be organized in BIDS format. Use tools like "dcm2bids" to convert DICOM to BIDS.
      - Install required python libraries with pip ("pip install -r requirements.txt").

    # ------------------------------------------------------------------------

    Modality-Specific QC Overview:

      Anatomical (anat):
        - Generates QC montages for all anatomical scans within the anat folder.

      Diffusion-Weighted Imaging (dwi):
        - Generates 4D QC montages, displaying one slice per DWI volume with separate scalings for each volume.
        - Sagittal view montages are useful for inspecting volumes corrupted by motion (e.g., venetian blind effect).
        - Axial views help assess the extent of EPI geometric distortions.
        - Automated reports are generated for motion corruption using 3dZipperZapper.
        - Calculates between-TR motion and generates corresponding motion plots.
        - Summarizes phase encoding information and slice-drop/motion corruption data.

      Functional (func):
        - Generates TSNR (Temporal Signal-to-Noise Ratio) and temporal standard deviation maps.
        - Calculates between-TR motion and generates motion plots and carpet grayplot images.
        - Calculates AFNI's outlier and quality indices.
        - Computes Ghost to Signal Ratio (GSR) in the x and y directions to quantify ghosting artifacts.
        - Detects multi-echo functional data (>= 3 echoes) and, if present, uses tedana to generate T2* and S0 maps, along with RMSE (Root Mean Square Error) maps.

    Summary:
      - Collects and summarizes key MRI acquisition parameters (e.g., matrix size, slice count, number of dynamics, TR, orientation) into an HTML table.
      - Provides a detailed, session-specific QC report in HTML format, offering a visual and quantitative assessment of the MRI data quality.

    # ------------------------------------------------------------------------

    Decision Buttons in QC Report:

      - After processing, the script generates a QC report in HTML format, which includes buttons for manual review of each file.
      - The buttons allow the user to categorize each file as:
          1. Reject/Problem
          2. Borderline/Warning
          3. OK
      - Users can also provide comments for each file.
      - A "Submit QC and Save Report" button at the end of the report allows saving the final decision for all files, including their QC status and comments, in a text file.

    # ------------------------------------------------------------------------
    # Fast MRI Quality Control (QC) Script v0.6
    # Written by Bharath Holla (NIMHANS, Bengaluru)
    # ------------------------------------------------------------------------


    """
    print(help_message)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fast MRI Quality Control (QC) Script")
    parser.add_argument("-p", "--subject_id", required=True, help="Subject ID to be processed.")
    parser.add_argument("-b", "--bids_root", default=os.getcwd(), help="Root directory of the BIDS dataset. Defaults to current directory.")
    parser.add_argument("-s", "--session", default="", help="Specific session to process.")
    parser.add_argument("-o", "--qc_root", default="", help="Output directory for QC reports. Defaults to 'qc' directory adjacent to the BIDS root.")
    args = parser.parse_args()
    return args

def check_subject_directory(bids_root, subject_id):
    subject_path = os.path.join(bids_root, f"sub-{subject_id}")
    if not os.path.isdir(subject_path):
        print(f"Error: Subject directory for sub-{subject_id} not found in {bids_root}.")
        sys.exit(1)
    return subject_path

def get_sessions_to_process(subject_path, session_id):
    """
    Determine sessions to process based on session_id and available sessions in subject path.
    If no session directories are found, it returns a list with an empty string to indicate subject-level processing.
    """
    session_dirs = [
        d for d in os.listdir(subject_path)
        if os.path.isdir(os.path.join(subject_path, d)) and d.startswith('ses-')
    ]

    if not session_dirs: # Check if session_dirs is empty (no session directories found)
        return [''] # Return a list with an empty string to indicate no sessions

    if session_id:
        if f"ses-{session_id}" in session_dirs:
            sessions_to_process = [f"ses-{session_id}"]
        else:
            print(f"Warning: Session ID 'ses-{session_id}' not found for subject. Processing all sessions.")
            sessions_to_process = session_dirs
    else:
        sessions_to_process = session_dirs

    return sessions_to_process


def create_qc_directories(bids_root, qc_root_arg):
    if not qc_root_arg:
        qc_root = os.path.join(os.path.dirname(bids_root), "qc") # adjacent to BIDS root
    else:
        qc_root = os.path.realpath(qc_root_arg) # Use realpath to resolve relative paths
    os.makedirs(qc_root, exist_ok=True) # exist_ok=True prevents error if directory exists
    return qc_root

def generate_html_report_header(html_report_path, subject_id, session_id):
    with open(html_report_path, 'w') as html_file: # 'w' mode will overwrite if file exists, consider 'x' for exclusive creation
        html_file.write("<html><head><title>MRI QC Report for {} {}</title></head><body>\n".format(subject_id, session_id)) # .format is good for readability, f-strings also great
        html_file.write("<style>img { max-width: 100%; height: auto; }</style>\n") # CSS in header, better to put in separate CSS file for larger reports
        html_file.write("<h1>MRI QC Report</h1>\n")
        html_file.write("<h2>Participant ID: {}</h2>\n".format(subject_id))
        html_file.write("<h2>Session ID: {}</h2>\n".format(session_id))

def generate_html_report_footer(html_report_path, duration): # Added duration argument
    with open(html_report_path, 'a') as html_file:
        # --- Add QC page generation duration at the end ---
        duration_minutes = int(duration.total_seconds() // 60) # Extract minutes
        duration_seconds = int(duration.total_seconds() % 60)  # Extract remaining seconds
        duration_string = f"{duration_minutes} minutes {duration_seconds} seconds" # Format duration

        html_file.write(f"<p style='font-size: small; font-style: italic;'>QC page generated in {duration_string}.</p>\n")
        html_file.write("</body>\n")
        html_file.write("</html>\n")
    return html_report_path

def nifti_info_json(file_path, prefix, modality):
    """
    Extracts NIfTI header information, writes it to a JSON file, and returns the extracted info.

    Combines the functionality of extract_nifti_info and write_nifti_info_json.
    Prioritizes TE extraction from NIfTI header's 'descrip' field, falls back to JSON sidecar.
    TR is extracted from pixdim[4] if not found in JSON.

    Args:
        file_path (str): Path to the NIfTI image file.
        prefix (str): Prefix for the output JSON file (e.g., path and base name).
        modality (str): Modality of the file (e.g., "anat", "func", "dwi").

    Returns:
        tuple: (success, nifti_info_dict)
               - success (bool): True if extraction and JSON writing were successful, False otherwise.
               - nifti_info_dict (dict or None): A dictionary containing the extracted NIfTI information
                 if successful, None otherwise. The dictionary has keys: 'dimensions', 'voxel_sizes',
                 'affine', 'orientation', 'tr', 'te'.
    """
    import nibabel as nib

    nifti_info = None 
    success = False 
    print(file_path)
    try:
        img = nib.load(file_path)
        header = img.header

        dimensions = img.shape
        voxel_sizes = img.header['pixdim'][1:4]

        affine = img.affine
        orientation = nib.orientations.aff2axcodes(affine)

        te = 'N/A'
        tr = 'N/A'

        # --- 1. Try to extract TE from descrip_field ---
        descrip_field = header.get('descrip')
        if descrip_field is not None:
            descrip_field_str = descrip_field.tobytes().decode()
            try:
                te = descrip_field_str.split("TE=")[1].split(";")[0]
                te = float(te) if te else 'N/A'
            except IndexError:
                print(f"      Warning: TE= not found in descrip_field for {os.path.basename(file_path)}. Falling back to JSON.")
                te = 'N/A'
            except Exception as e:
                print(f"      Warning: Error extracting TE from descrip field for {os.path.basename(file_path)}: {e}. Falling back to JSON.")
                te = 'N/A'
        else:
            print(f"      Warning: descrip_field not found in header for {os.path.basename(file_path)}. Falling back to JSON.")

        # --- 2. Fallback to JSON sidecar file if TE is still 'N/A' ---
        if te == 'N/A':
            json_file = file_path.replace('.nii.gz', '.json')
            try:
                if os.path.exists(json_file):
                    with open(json_file, 'r') as f:
                        json_data = json.load(f)
                        te_json = json_data.get('EchoTime', 'N/A')
                        tr_json = json_data.get('RepetitionTime', 'N/A')
                        te = float(te_json) if te_json != 'N/A' else 'N/A'
                        tr = float(tr_json) if tr_json != 'N/A' else 'N/A'
                else:
                    print(f"      Warning: JSON sidecar file not found: {json_file}")
            except json.JSONDecodeError:
                print(f"      Warning: Could not decode JSON file: {json_file}")
            except Exception as e:
                print(f"      Warning: Error extracting TR/TE from JSON {json_file}: {e}")

        # --- 3. Robust TR extraction from pixdim[4] (NIfTI header), used if JSON TR not found ---
        if tr == 'N/A':
            tr_pixdim = img.header['pixdim'][4]
            tr = tr_pixdim if tr_pixdim > 0 else 'N/A'

        nifti_info = {
            'dimensions': dimensions,
            'voxel_sizes': voxel_sizes,
            'affine': affine,
            'orientation': orientation,
            'tr': tr,
            'te': te,
        }

        info_dict = {
            "MatrixSize_x": int(dimensions[0]),
            "MatrixSize_y": int(dimensions[1]),
            "MatrixSize_z": int(dimensions[2]),
            "Volumes": int(dimensions[3] if len(dimensions) > 3 else 1),
            "PixelDimension_i": float(voxel_sizes[0]),
            "PixelDimension_j": float(voxel_sizes[1]),
            "SliceThickness_k": float(voxel_sizes[2]),
            "TR": float(tr) if isinstance(tr, (int, float)) else str(tr),
            "TE": float(te) if isinstance(te, (int, float)) else str(te),
            "Orientation": str(orientation),
            "Filename": os.path.basename(file_path)
        }

        json_file_path = f"{prefix}_info.json"
        with open(json_file_path, "w") as json_file:
            json.dump(info_dict, json_file, indent=4)
        success = True # Set success to True if JSON writing is reached (and no earlier exceptions)
        return success, nifti_info # Return success status and nifti_info

    except Exception as e:
        print(f"      Warning: Error in extract_and_write_nifti_info_json for {os.path.basename(file_path)}: {e}, Exception Type: {type(e).__name__}, Error: {e}")
        return False, None # Indicate failure with False and return None for nifti_info



def process_anatomical_modality(session_path, qc_session_path, html_report_path):
    print("--- process_anatomical_modality function START ---")
    modality = "anat"
    mod_path = os.path.join(session_path, modality)
    if not os.path.isdir(mod_path):
        print(f"  Modality path {mod_path} does not exist. Skipping anat.")
        print("--- process_anatomical_modality function END (skipped) ---")
        return

    with open(html_report_path, 'a') as html_file:
        html_file.write("<h2>{}</h2>\n".format(modality))

    nii_gz_files = glob.glob(os.path.join(mod_path, "*.nii.gz"))
    for file_path in nii_gz_files:
        base_name = os.path.basename(file_path).replace(".nii.gz", "")
        json_path = os.path.join(mod_path, base_name + ".json")
        prefix = os.path.join(qc_session_path, f"{base_name}_{modality}")
        print(f"  {base_name}")

        success, _ = nifti_info_json(file_path, prefix, modality)
        if not success:
            print(f"      Warning: JSON writing failed for {base_name}, skipping HTML report table for this file.")
            continue # Skip to next file if JSON writing failed

        try:
            from nilearn import plotting
            import matplotlib.pyplot as plt  # Keep matplotlib for figure handling
            import nibabel as nib
            
            img = nib.load(file_path)
            img_data = img.get_fdata() # Get the image data as a numpy array
            vmin_percentile = np.percentile(img_data, 2) # 2nd percentile
            vmax_percentile = np.percentile(img_data, 98) # 98th percentile
            
            fig = plt.figure(figsize=(12, 12), dpi=200)  # Create a figure object, but do NOT pass to plot_stat_map
            plotting.plot_stat_map(
                stat_map_img=file_path, 
                bg_img=None, 
                display_mode='mosaic',
                cut_coords=None,
                title=base_name,
                annotate=False,
                draw_cross=False,
                black_bg=True, 
                cmap='gray',
                colorbar=False, 
                vmin=vmin_percentile, 
                vmax=vmax_percentile  
            )

            # --- Convert the Matplotlib figure to Base64 encoded PNG ---
            img_buf = io.BytesIO()
            plt.gcf().savefig(img_buf, format='png', dpi=200, bbox_inches='tight', pad_inches=0) # NEW: save current figure using plt.gcf()
            img_buf.seek(0)
            img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
            plt.close(fig) 

            # --- Create the data: URL for embedding ---
            data_url = f"data:image/png;base64,{img_base64}"

            with open(html_report_path, 'a') as html_file:
                html_file.write("<h4>{} </h4><img src='{}' alt='Montage for {}' style='width: 100%;'>\n".format(base_name, data_url, base_name)) # Use data_url in src

        except Exception as e:
            print(f"      Warning: Could not generate and embed Nilearn anatomical montage for {base_name}: {e}, Exception: {e}")
            with open(html_report_path, 'a') as html_file:
                html_file.write("<h4>{} </h4><p>Nilearn anatomical montage embedding failed.</p>\n".format(base_name))
    print("--- process_anatomical_modality function END ---")


def process_dwi_modality(session_path, qc_session_path, html_report_path):
    modality = "dwi"
    mod_path = os.path.join(session_path, modality)
    if not os.path.isdir(mod_path):
        return

    with open(html_report_path, 'a') as html_file:
        html_file.write("<h2>{}</h2>\n".format(modality))

    nii_gz_files = glob.glob(os.path.join(mod_path, "*.nii.gz"))
    for file_path in nii_gz_files:
        base_name = os.path.basename(file_path).replace(".nii.gz", "")
        json_path = os.path.join(mod_path, base_name + ".json")
        prefix = os.path.join(qc_session_path, f"{base_name}_{modality}")

        if "_echo-1_" in base_name or "_echo-3_" in base_name:
            continue

        print(f"  {base_name}")

        success, _ = nifti_info_json(file_path, prefix, modality)
        if not success:
            print(f"      Warning: JSON writing failed for {base_name}, skipping HTML report table for this file.")
            continue # Skip to next file if JSON writing failed
        
        import subprocess  

        # --- Run 3dAutomask ---
        automask_prefix = f"{prefix}_mask.nii.gz"
        automask_command = ["3dAutomask", "-overwrite", "-prefix", automask_prefix, f"{file_path}[0]"]
        try:
            subprocess.run(automask_command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"     Warning: 3dAutomask failed for {base_name}: {e}")

        # --- Run 3dZipperZapper ---
        zz_prefix = f"{prefix}_zz"
        zipperzapper_command = ["3dZipperZapper", "-prefix", zz_prefix, "-input", file_path, "-mask", automask_prefix, "-no_out_bad_mask"]
        try:
            subprocess.run(zipperzapper_command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"     Warning: 3dZipperZapper failed for {base_name}: {e}")


        try:
            from nilearn import plotting, image
            import matplotlib.pyplot as plt
            import numpy as np
            import io
            import base64
            import math  
            import nibabel as nib

            # --- Load DWI image ---
            img = nib.load(file_path)
            data = img.get_fdata()  # shape (x, y, z, volumes)
            num_volumes = data.shape[3] if len(data.shape) > 3 else 1

            # --- Calculate middle axial slice index ---
            mid_sag = data.shape[1] // 10
            mid_cor = data.shape[1] // 10
            mid_axi = data.shape[2] // 2

            # --- Calculate grid layout parameters ---
            images_per_row = 7
            num_rows = math.ceil(num_volumes / images_per_row)  # Ceiling to ensure all volumes are included

            # --- Generate Nilearn plots for ALL MIDDLE AXIAL SLICES - SEP-SCL, GRID LAYOUT ---
            montage_base64_list = []  # List to store base64 encoded montages
            for vol_index, volume_img in enumerate(image.iter_img(img)):  # Iterate through volumes using iter_img
                # --- Calculate vmin and vmax for CONTRAST ADJUSTMENT - VOLUME-SPECIFIC ---
                volume_data = volume_img.get_fdata()  # Get data for the current 3D volume
                vmin_percentile = np.percentile(volume_data, 15)  # 2nd percentile for THIS volume
                vmax_percentile = np.percentile(volume_data, 99)  # 98th percentile for THIS volume

                # --- Generate Nilearn plot for middle axial slice using plot_stat_map ---
                fig = plt.figure(figsize=(6, 6), dpi=150)
                plt.rcParams['figure.max_open_warning'] = 100  # Increase the threshold, or set to 'off' to disable
                plotting.plot_stat_map(
                    volume_img,  # Plot the 3D volume directly from iter_img
                    bg_img=None,
                    #cut_coords=[mid_sag,mid_cor,mid_axi],  # Middle slices for this volume
                    cut_coords=None,
                    #display_mode='ortho',
                    title=f'DWI Vol {vol_index}',  
                    black_bg=True,
                    annotate=True,
                    draw_cross=False,
                    cmap='gray',
                    colorbar=False,
                    vmin=vmin_percentile,  # VOLUME-SPECIFIC vmin
                    vmax=vmax_percentile  # VOLUME-SPECIFIC vmax
                )

                # --- Convert the Matplotlib figure to Base64 encoded PNG ---
                img_buf = io.BytesIO()
                plt.gcf().savefig(img_buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0)
                img_buf.seek(0)
                img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
                plt.close(fig)
                montage_base64_list.append(img_base64)

            # --- Write HTML for montage with SEPARATE SCALING - GRID LAYOUT ---
            with open(html_report_path, 'a') as html_file:
                html_file.write("<h4> {} </h4>\n".format(base_name))  # Updated title for grid layout

                for row_index in range(num_rows):  # Iterate through rows
                    html_file.write("<div style='display: flex; flex-direction: row; justify-content: flex-start; margin-bottom: 10px;'>\n")  # Flex container for each row
                    for col_index in range(images_per_row):  # Iterate through columns in each row
                        vol_index = row_index * images_per_row + col_index  # Calculate volume index
                        if vol_index < num_volumes:  # Check if volume index is within range
                            img_base64 = montage_base64_list[vol_index]
                            data_url = f"data:image/png;base64,{img_base64}"
                            html_file.write("<div style='width: {}px; margin-right: 5px;'>".format(180))  # Adjust width and right margin
                            html_file.write("<img src='{}' alt='Sep-Scl Montage Vol {} for {}' style='width: 100%;'>\n".format(data_url, vol_index, base_name))
                            html_file.write("</div>\n")
                    html_file.write("</div>\n")  # Close row flex container

        except Exception as e:
            print(f"    Warning: Could not generate and embed Nilearn DWI montage (middle axial slices, sep-scl, grid layout) for {base_name}: {e}, Exception: {e}")
            with open(html_report_path, 'a') as html_file:
                html_file.write("<h4>DWI Montage - Middle Axial Slice Across Volumes (Sep-Scl, Grid Layout) {} </h4><p>Nilearn DWI montage (middle axial slices, sep-scl, grid layout) generation failed.</p>\n".format(base_name))


        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from nipype.interfaces import afni
            import io
            import base64
            import tempfile
            import pandas as pd  
            import subprocess  

            # --- Create a temporary file for out_file ---
            temp_out_file = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=True)
            motion_file_1D = os.path.join(qc_session_path, f"dfile_rall_{base_name}.1D")

            volreg = afni.Volreg()
            volreg.inputs.in_file = file_path
            volreg.inputs.out_file = temp_out_file.name  # Use temporary file
            volreg.inputs.oned_file = motion_file_1D
            volreg.inputs.oned_matrix_save = os.path.join(qc_session_path, f"mat_{base_name}.1D")
            volreg.inputs.md1d_file = os.path.join(qc_session_path, f"max_{base_name}.1D") # Correct Path
            volreg.inputs.interp = 'cubic'
            volreg.inputs.zpad = 4
            volreg.inputs.verbose = False
            volreg.inputs.basefile = file_path

            volreg_result = volreg.run()
            temp_out_file.close()  # Close temporary file

            # --- Calculate Enorm using 1d_tool.py shell command via subprocess ---
            enorm_file_1D = os.path.join(qc_session_path, f"motion_{base_name}_enorm.1D")
            command_1dtool = [
                "1d_tool.py",
                "-infile", motion_file_1D,
                "-set_nruns", "1",
                "-derivative",
                "-collapse_cols", "weighted_enorm",
                "-weight_vec", ".9", ".9", ".9", "1", "1", "1",
                "-write", enorm_file_1D
            ]
            subprocess.run(command_1dtool, check=True)  

            num_volumes = sum(1 for line in open(enorm_file_1D))
            if num_volumes > 9:
                motion_plot_prefix = os.path.join(qc_session_path, f"motion_outlier_plot_{base_name}")
                motion_file_1D = os.path.join(qc_session_path, f"dfile_rall_{base_name}.1D")
                enorm_file_1D = os.path.join(qc_session_path, f"motion_{base_name}_enorm.1D")

                command_1dplot = [
                    "1dplot.py",
                    "-sepscl",
                    "-boxplot_on",
                    "-reverse_order",
                    "-infiles", motion_file_1D, enorm_file_1D,
                    "-ylabels", "VOLREG", "enorm",
                    "-xlabel", "vols",
                    "-title", f"Motion Profile: {base_name}",  
                    "-prefix", motion_plot_prefix
                ]
                subprocess.run(command_1dplot, check=True)  

                # --- NEW: Load the generated PNG, convert to Base64, and prepare for HTML embedding ---
                motion_plot_jpg = motion_plot_prefix + ".jpg"  
                motion_plot_base64 = None  

                if os.path.exists(motion_plot_jpg):  
                    try:
                        with open(motion_plot_jpg, 'rb') as img_file:  
                            img_buf = io.BytesIO(img_file.read())  
                        motion_plot_base64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')  
                    except Exception as embed_e:
                        print(f"      Warning: Could not read/encode generated motion plot {motion_plot_jpg} for HTML embedding: {embed_e}")
                        motion_plot_base64 = None  # Ensure it's None if encoding fails
                else:
                    print(f"      Warning: Motion plot JPG file {motion_plot_jpg} was not found after 1dplot.py execution.")

                # --- Embed motion plot in HTML ---
                with open(html_report_path, 'a') as html_file:
                    if motion_plot_base64 is not None:  # Embed plot only if generated and encoded successfully
                        data_url = f"data:image/png;base64,{motion_plot_base64}"
                        html_file.write("<h4>Motion Profile - {}</h4><img src='{}' alt='Motion Profile Style {}' style='width: 80%;'>\n".format(base_name, data_url, base_name))
                    else:
                        print(f"      [DEBUG] motion_plot_base64 is None, skipping HTML embedding for: {base_name}")  # DEBUG PRINT - HTML EMBEDDING SKIPPED

        except Exception as e:
            with open(html_report_path, 'a') as html_file:
                if motion_plot_base64 is None:  
                    html_file.write("<p>afni 1dplot style motion plot generation failed.</p>\n")

    return html_report_path


def process_functional_modality(session_path, qc_session_path, html_report_path):
    modality = "func"
    mod_path = os.path.join(session_path, modality)
    if not os.path.isdir(mod_path):
        return

    with open(html_report_path, 'a') as html_file:
        html_file.write("<h2>{}</h2>\n".format(modality))

    nii_gz_files = glob.glob(os.path.join(mod_path, "*.nii.gz"))
    for file_path in nii_gz_files:
        base_name = os.path.basename(file_path).replace(".nii.gz", "")
        json_path = os.path.join(mod_path, base_name + ".json")
        prefix = os.path.join(qc_session_path, f"{base_name}_{modality}")

        success, _ = nifti_info_json(file_path, prefix, modality)
        if not success:
            print(f"      Warning: JSON writing failed for {base_name}, skipping HTML report table for this file.")
            continue # Skip to next file if JSON writing failed

        if "_echo-1_" in base_name or "_echo-3_" in base_name:
            continue

        if "sbref" in base_name: 
            continue    

        print(f"  {base_name}")

        # --- Replace 3dvolreg & 1d_tool.py & 1dplot.py (for motion) ---
        censor_file = None
        motion_plot_base64 = None
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from nipype.interfaces import afni
            import io
            import base64
            import tempfile
            import pandas as pd
            import subprocess

            temp_out_file = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=True, dir=qc_session_path) # Explicit dir
            motion_file_1D = os.path.join(qc_session_path, f"dfile_rall_{base_name}.1D")

            volreg = afni.Volreg()
            volreg.inputs.in_file = file_path
            volreg.inputs.out_file = temp_out_file.name
            volreg.inputs.oned_file = motion_file_1D
            volreg.inputs.oned_matrix_save = os.path.join(qc_session_path, f"mat_{base_name}.1D") # Correct Path
            volreg.inputs.md1d_file = os.path.join(qc_session_path, f"max_{base_name}.1D") # Correct Path
            volreg.inputs.interp = 'cubic'
            volreg.inputs.zpad = 4
            volreg.inputs.verbose = False
            volreg.inputs.basefile = file_path

            volreg_result = volreg.run()
            temp_out_file.close()

            enorm_file_1D = os.path.join(qc_session_path, f"motion_{base_name}_enorm.1D")
            command_1dtool = [
                "1d_tool.py",
                "-infile", motion_file_1D,
                "-set_nruns", "1",
                "-derivative",
                "-collapse_cols", "weighted_enorm",
                "-weight_vec", ".9", ".9", ".9", "1", "1", "1",
                "-write", enorm_file_1D
            ]
            subprocess.run(command_1dtool, check=True)
            

            censor_file = os.path.join(qc_session_path, f"{base_name}") # Define censor file path
            try:
                import subprocess # Import subprocess for running command-line tools

                command_1dtool_censor = [ # Define command for 1d_tool.py to generate censor file
                    "1d_tool.py",
                    "-infile", motion_file_1D,
                    "-set_nruns", "1",
                    "-censor_prev_TR", # Flag for censoring previous TR
                    "-censor_motion", "0.3", # Motion threshold argument
                     censor_file  # Output censor file path
                ]
                print(f"DEBUG: 1d_tool.py command: {' '.join(command_1dtool_censor)}") # Print command

                subprocess.run(command_1dtool_censor, check=True) # Execute the 1d_tool.py command
                censor_file_1D = os.path.join(qc_session_path, f"{base_name}_censor.1D") 
            except Exception as e:
                censor_file = None # If censor file generation fails, set to None
                print(f"    Warning: Could not generate censor file using 1d_tool.py for {base_name}: {e}")

            fwd_file_1D = os.path.join(qc_session_path, f"fwd_{base_name}_abssum.1D") # Define FWD file path
            try:
                from nipype.interfaces import afni # Import nipype afni interface

                tstat_fwd = afni.TStat() # Initialize nipype TStat interface
                tstat_fwd.inputs.in_file = enorm_file_1D # Input Enorm file
                tstat_fwd.inputs.args = '-abssum'      # Use -abssum argument for FWD
                tstat_fwd.inputs.out_file = fwd_file_1D  # Output FWD file path
                tstat_fwd_result = tstat_fwd.run()       # Execute 3dTstat via Nipype

            except Exception as e:
                fwd_file_1D = None # If FWD calculation fails, set to None
                print(f"    Warning: Could not generate FWD file using nipype afni.TStat for {base_name}: {e}")


            tqual_file_1D = os.path.join(qc_session_path, f"3dTqual_{base_name}_range.1D") # Define TQual file path
            try:
                from nipype.interfaces import afni # Ensure afni interface is imported

                quality_index = afni.QualityIndex() # Initialize nipype QualityIndex interface
                quality_index.inputs.in_file = file_path # Input functional NIfTI file
                quality_index.inputs.automask = True     # Enable automask option
                quality_index.inputs.spearman = True     # Enable spearman correlation option
                quality_index.inputs.out_file = tqual_file_1D # Output TQual file path
                tqual_result = quality_index.run()        # Execute 3dTqual via Nipype

            except Exception as e:
                tqual_file_1D = None # If TQual calculation fails, set to None
                print(f"    Warning: Could not generate TQual file using nipype afni.QualityIndex for {base_name}: {e}")


            # --- Outlier Count Fraction Calculation (replace shell script with nipype afni.OutlierCount) ---
            outlier_file_1D = os.path.join(qc_session_path, f"3dToutcount_fraction_{base_name}.1D") # Define Outlier file path
            outlier_nii_gz = os.path.join(qc_session_path, f"{base_name}_outlier_{modality}.nii.gz") # Define outlier NIfTI output path

            try:
                from nipype.interfaces import afni # Ensure afni interface is imported

                outlier_count = afni.OutlierCount() # Initialize nipype OutlierCount interface
                outlier_count.inputs.in_file = file_path # Input functional NIfTI file
                outlier_count.inputs.automask = True      # Enable automask option
                outlier_count.inputs.fraction = True      # Calculate fraction of outliers
                outlier_count.inputs.legendre = True      # Use Legendre polynomials option
                outlier_count.inputs.outliers_file = outlier_nii_gz # Save outlier volume as NIfTI
                outlier_count.inputs.out_file = outlier_file_1D # Output Outlier count file path
                outlier_result = outlier_count.run()        # Execute 3dToutcount via Nipype

            except Exception as e:
                outlier_file_1D = None # If Outlier calculation fails, set to None
                print(f"    Warning: Could not generate Outlier count file using nipype afni.OutlierCount for {base_name}: {e}")

            num_volumes = sum(1 for line in open(enorm_file_1D))
            if num_volumes > 9:
                motion_plot_prefix = os.path.join(qc_session_path, f"motion_outlier_plot_{base_name}")
                motion_file_1D = os.path.join(qc_session_path, f"dfile_rall_{base_name}.1D")
                enorm_file_1D = os.path.join(qc_session_path, f"motion_{base_name}_enorm.1D")

                command_1dplot = [
                    "1dplot.py",
                    "-sepscl",
                    "-boxplot_on",
                    "-reverse_order",
                    "-censor_files", censor_file_1D, # Add censor file as input
                    "-infiles", motion_file_1D, enorm_file_1D, outlier_file_1D, tqual_file_1D, # Include outlier_file_1D
                    "-ylabels", "VOLREG", "enorm", "aoi","aqi", # Add 'outliers' ylabel
                    "-xlabel", "vols",
                    "-censor_hline", "0.3", "0.3", "0.3", "0.3", "0.3", "0.3", "0.3", "0.05","0.01", # Censor threshold lines (motion and outlier fraction)
                    "-title", f"Motion Profile: {base_name}",
                    "-prefix", motion_plot_prefix
                ]
                subprocess.run(command_1dplot, check=True)

                motion_plot_jpg = motion_plot_prefix + ".jpg"
                motion_plot_base64 = None

                if os.path.exists(motion_plot_jpg):
                    try:
                        with open(motion_plot_jpg, 'rb') as img_file:
                            img_buf = io.BytesIO(img_file.read())
                        motion_plot_base64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')
                    except Exception as embed_e:
                        print(f"      Warning: Could not read/encode generated motion plot {motion_plot_jpg} for HTML embedding: {embed_e}")
                        motion_plot_base64 = None
                else:
                    print(f"      Warning: Motion plot JPG file {motion_plot_jpg} was not found after 1dplot.py execution.")

                with open(html_report_path, 'a') as html_file:
                    if motion_plot_base64 is not None:
                        data_url = f"data:image/png;base64,{motion_plot_base64}"
                        html_file.write("<h4>Motion Profile - {}</h4><img src='{}' alt='Motion Profile Style {}' style='width: 80%;'>\n".format(base_name, data_url, base_name))


        except Exception as e:
            print(f"    Warning: Could not generate MRIQC-style motion plots for {base_name} or run OneDToolPy or 1dplot: {e}")
            with open(html_report_path, 'a') as html_file:
                html_file.write("<h4>Motion Profile -  Style {}</h4><p> motion profile generation failed.</p>\n".format(base_name))



        # TSNR & TSTD with plot_epi
        try:
            import nibabel as nib
            import numpy as np
            import matplotlib.pyplot as plt
            from nilearn import plotting 
            import io
            import base64
            from nipype.interfaces import afni 


            img = nib.load(file_path)
            data = img.get_fdata()

            # --- Calculate TSNR using nipype afni.TStat ---
            tsnr_prefix = f"{prefix}_tsnr" # Define prefix for TSNR output files
            tstat_tsnr = afni.TStat()
            tstat_tsnr.inputs.in_file = file_path
            tstat_tsnr.inputs.args = '-cvarinv' # Use -cvarinv option for TSNR-like measure
            tstat_tsnr.inputs.out_file = tsnr_prefix + ".nii.gz" # Explicitly set output file name
            tsnr_result = tstat_tsnr.run()
            tsnr_nii_path = tsnr_result.outputs.out_file # Get the output file path from nipype

            tsnr_img = nib.load(tsnr_nii_path) # Load the TSNR image generated by 3dTstat
            tsnr_data = tsnr_img.get_fdata() # Get TSNR data for mask
            vmin_percentile = np.percentile(tsnr_data, 2) # 2nd percentile
            vmax_percentile = np.percentile(tsnr_data, 98) # 98th percentile
            # --- Nilearn plot_epi montage for TSNR ---
            fig_tsnr_epi = plt.figure(figsize=(10, 4),dpi=150) # Create figure for TSNR plot_epi
            plotting.plot_stat_map(
                tsnr_nii_path,
                bg_img=None,
                display_mode='mosaic',
                annotate=False,
                cut_coords=None,
                draw_cross=False,
                cmap='gray',
                black_bg=True,
                colorbar=False,
                vmin=vmin_percentile,
                vmax=vmax_percentile,
                figure=fig_tsnr_epi, 
                title=f'TSNR - {base_name}' 
            )

            img_buf_tsnr_epi = io.BytesIO()
            fig_tsnr_epi.savefig(img_buf_tsnr_epi, format='png', bbox_inches='tight', pad_inches=0)
            img_buf_tsnr_epi.seek(0)
            tsnr_montage_base64 = base64.b64encode(img_buf_tsnr_epi.read()).decode('utf-8')
            plt.close(fig_tsnr_epi)


            with open(html_report_path, 'a') as html_file:
                html_file.write(f"<h4>Temporal signal to noise ratio (TSNR) - {base_name}</h4><img src='data:image/png;base64,{tsnr_montage_base64}' alt='TSNR plot_epi Montage {base_name}' style='width: 80%;'>\n")


            # --- Calculate TSTD using nipype afni.TStat ---
            tstd_prefix = f"{prefix}_tstd" # Define prefix for TSTD output files
            tstat_tstd = afni.TStat()
            tstat_tstd.inputs.in_file = file_path
            tstat_tstd.inputs.args = '-stdev' # Use -stdev option for temporal std dev
            tstat_tstd.inputs.out_file = tstd_prefix + ".nii.gz" # Explicitly set output file name
            tstd_result = tstat_tstd.run()
            tstd_nii_path = tstd_result.outputs.out_file # Get the output file path from nipype


            tstd_data = nib.load(tstd_nii_path).get_fdata()
            vmin_percentile = np.percentile(tstd_data, 2) # 2nd percentile
            vmax_percentile = np.percentile(tstd_data, 98) # 98th percentile
            
            # --- Nilearn plot_epi montage for TSTD ---
            fig_tstd_epi = plt.figure(figsize=(10, 4),dpi=150) # Create figure for TSTD plot_epi
            plotting.plot_stat_map(
                tstd_nii_path,
                bg_img=None,
                display_mode='mosaic',
                annotate=False,
                cut_coords=None,
                draw_cross=False,
                cmap='gray',
                black_bg=True,
                colorbar=False,
                vmin=vmin_percentile,
                vmax=vmax_percentile,
                figure=fig_tstd_epi, 
                title=f'TSTD - {base_name}' 
            )

            img_buf_tstd_epi = io.BytesIO()
            fig_tstd_epi.savefig(img_buf_tstd_epi, format='png', bbox_inches='tight', pad_inches=0)
            img_buf_tstd_epi.seek(0)
            tstd_montage_base64 = base64.b64encode(img_buf_tstd_epi.read()).decode('utf-8')
            plt.close(fig_tstd_epi)


            with open(html_report_path, 'a') as html_file:
                html_file.write(f"<h4>Temporal standard deviation (TSTD) - {base_name}</h4><img src='data:image/png;base64,{tstd_montage_base64}' alt='TSTD plot_epi Montage {base_name}' style='width: 80%;'>\n")

        except Exception as e:
            print(f"    Warning: Could not generate TSNR/TSTD plot_epi montages for {base_name}: {e}")
            with open(html_report_path, 'a') as html_file:
                html_file.write("<h4>TSNR/TSTD plot_epi montage generation failed for {}.</h4>\n".format(base_name))

                print(f"  {base_name}")

    return html_report_path



def process_mefmri_modality(session_path, qc_session_path, html_report_path):
    """
    Processes multi-echo fMRI data to generate T2*, S0, and RMSE maps and plots for each task and run.
    Processes each task and run separately using t2smap.

    Args:
        session_path (str): Path to the session directory.
        qc_session_path (str): Path to the QC output directory.
        html_report_path (str): Path to the HTML report file.
    """
    import json
    import subprocess
    import nibabel as nib
    import numpy as np
    import matplotlib.pyplot as plt
    from nilearn import plotting
    import io
    import base64
    from nipype.interfaces import afni 
    import tempfile
    import pandas as pd
    import glob
    import os

    modality = "func"
    mod_path = os.path.join(session_path, modality)

    if not os.path.isdir(mod_path):
        return

    with open(html_report_path, 'a') as html_file:
            # Multi-echo Functional - By Task and Run Section
            nii_gz_files = glob.glob(os.path.join(mod_path, "*_echo-*.nii.gz")) # Look for multi-echo files
            print(f"    possible_echo_files :{nii_gz_files}")
            if not nii_gz_files:
                print("    No multi-echo files found, skipping mefMRI processing and section.")
                # Do NOT write heading, just return and skip the section
                return # Skip the rest of the ME-fMRI processing for this session
            else:
                html_file.write("<h2>Multi-echo Functional MRI</h2>\n") # Only write heading if ME files exist


    # --- Determine robust task_base from all possible_echo_files (still needed for common parts) ---
    example_filename = os.path.basename(nii_gz_files[0])
    parts = example_filename.split('_')
    task_base_parts = []
    for part in parts:
        if "run-" not in part and "echo-" not in part and "task-" in part: # Keep "task-" part for task grouping, exclude run/echo
            task_base_parts.append(part)
    task_base = "_".join(task_base_parts) # Reconstruct task_base (now task-specific base)

    print(f"    Task base: {task_base}") # General task base across runs


    # --- Group files by task and run ---
    task_run_file_groups = {}
    for file_path in nii_gz_files:
        base_name = os.path.basename(file_path).replace(".nii.gz", "")
        parts = base_name.split('_')
        task_name = "unknown_task"  # Default task name
        run_num = "unknown_run"     # Default run number
        echo_num_str = "unknown_echo"

        for part in parts: # Extract task and run and echo numbers
            if "task-" in part:
                task_name = part
            if "run-" in part:
                run_num = part
            if "echo-" in part:
                echo_num_str = part # Keep echo part for sorting later


        task_run_key = (task_name, run_num) # Create a tuple key for task and run
        if task_run_key not in task_run_file_groups:
            task_run_file_groups[task_run_key] = []
        task_run_file_groups[task_run_key].append({'file_path': file_path, 'base_name': base_name, 'echo_num_str': echo_num_str}) # Store file info

    sorted_task_run_keys = sorted(task_run_file_groups.keys(), key=lambda key: int(key[1].split('-')[1])) # Sort by run number
    print(f"    Task (Run): {sorted_task_run_keys}") # Print sorted task/run groups

    for task_run_key in sorted_task_run_keys: # Iterate through sorted keys
        files_info = task_run_file_groups[task_run_key]
        task_name, run_num = task_run_key
        task_run_base_name = f"{task_name}_{run_num}" # Create a base name for this task and run
        qc_task_run_session_path = os.path.join(qc_session_path, task_run_base_name) # QC output path for task/run
        os.makedirs(qc_task_run_session_path, exist_ok=True) # Create task/run QC dir

        print(f"    Processing Task: {task_name}, Run: {run_num}")

        # Sort files within each task/run group by echo number
        files_info.sort(key=lambda item: int(item['echo_num_str'].split('-')[1])) # Sort by echo number, though might not be needed for t2smap approach

        echo_files_for_run = [f_info['file_path'] for f_info in files_info] # Extract file paths for current task/run
        if len(echo_files_for_run) < 2: # Require at least 2 echoes to process
            print(f"    Warning: Not enough echoes (less than 2) for Task: {task_name}, Run: {run_num}, skipping t2smap.")
            html_file.write(f"<p>Warning: Not enough echoes (less than 2) for Task: {task_name}, Run: {run_num}, skipping t2smap.</p>\n")
            continue # Skip to the next task/run group

        # --- Echo time extraction and t2smap processing for this task/run ---
        echo_times = []
        absolute_echo_files = [] # List to store absolute paths

        for echo_file_path in echo_files_for_run:
            echo_base_name = os.path.basename(echo_file_path).replace(".nii.gz", "")
            json_path = os.path.join(mod_path, echo_base_name + ".json")

            echo_time = None # Initialize echo_time for each file
            try:
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                    echo_time = json_data.get('EchoTime') # Extract EchoTime from JSON
            except:
                print(f"      Warning: Could not load or extract EchoTime from JSON for {echo_base_name} (Task: {task_name}, Run: {run_num})")

            if echo_time is not None:
                echo_times.append(float(echo_time) * 1000) # Convert to milliseconds and append
                absolute_echo_files.append(os.path.abspath(echo_file_path)) # Get absolute path and store
            else:
                echo_times.append(None) # Append None if echo time extraction failed
                absolute_echo_files.append(os.path.abspath(echo_file_path)) # Still add absolute path for consistency in file list


        if not all(et is not None for et in echo_times):
            with open(html_report_path, "a") as html_report: # Still writing to main report, warnings could be in task/run sections too
                html_file.write(f"<p>Warning: Could not extract all echo times for {task_run_base_name}. Multi-echo processing may be incomplete for this Task/Run.</p>\n")
            print(f"      Warning: Incomplete echo times for {task_run_base_name}, skipping ME-fMRI processing for this Task/Run.")
            continue # Skip t2smap for this task/run if echo times are missing

        # Run t2smap for this task and run
        task_run_prefix = os.path.join(qc_task_run_session_path, task_run_base_name) # Prefix now includes task_run_base_name

        echo_files_str = " ".join([f'"{f}"' for f in absolute_echo_files]) # Double quotes for file paths in shell
        echo_times_str = " ".join(map(str, echo_times))

        t2smap_cmd_str = f"t2smap -d {echo_files_str} -e {echo_times_str} --out-dir {qc_task_run_session_path} --prefix {task_run_base_name}" # Prefix and out-dir specific to task/run

        script_file = os.path.join(qc_task_run_session_path, f"run_t2smap_{task_run_base_name}.sh") # Script file in QC task/run dir

        with open(script_file, 'w') as sh_file:
            sh_file.write("#!/bin/bash\n")
            sh_file.write(t2smap_cmd_str + "\n")
        os.chmod(script_file, 0o755) # Make script executable

        try:
            script_name = os.path.basename(script_file)
            absolute_script_path = os.path.join(qc_task_run_session_path, script_name)
            subprocess.run([absolute_script_path], check=True)


            # --- Nilearn plot_stat_map montage for T2* map ---
            t2star_nii_path = os.path.join(qc_task_run_session_path, f"{task_run_base_name}_T2starmap.nii.gz") # Path for task/run
            print(t2star_nii_path)
            t2star_img = nib.load(t2star_nii_path)
            t2star_data = t2star_img.get_fdata()
            vmin_percentile = np.percentile(t2star_data, 2)
            vmax_percentile = np.percentile(t2star_data, 99)

            fig_t2star_epi = plt.figure(figsize=(10, 4), dpi=150)
            plotting.plot_stat_map(
                t2star_nii_path,
                bg_img=None,
                display_mode='mosaic',
                annotate=False,
                cut_coords=None,
                draw_cross=False,
                cmap='gray', # Changed cmap for T2*
                black_bg=False, # Changed black_bg to False for T2*
                colorbar=True, # Keep colorbar for maps
                vmin=vmin_percentile,
                vmax=vmax_percentile,
                figure=fig_t2star_epi,
                title=f'T2* map - {task_name}, Run: {run_num}' # Title updated for task/run
            )
            img_buf_t2star_epi = io.BytesIO()
            fig_t2star_epi.savefig(img_buf_t2star_epi, format='png', bbox_inches='tight', pad_inches=0)
            img_buf_t2star_epi.seek(0)
            t2star_montage_base64 = base64.b64encode(img_buf_t2star_epi.read()).decode('utf-8')
            plt.close(fig_t2star_epi)

            with open(html_report_path, "a") as html_report:
                html_report.write(f"<h4>T2* map - {task_name}, Run: {run_num}</h4><img src='data:image/png;base64,{t2star_montage_base64}' alt='T2* map plot_stat_map Montage {task_run_base_name}' style='width: 80%;'>\n") # Alt text updated


            # --- Nilearn plot_stat_map montage for S0 map ---
            s0map_nii_path = os.path.join(qc_task_run_session_path, f"{task_run_base_name}_S0map.nii.gz") # Path for task/run
            s0map_img = nib.load(s0map_nii_path)
            s0map_data = s0map_img.get_fdata()
            vmin_percentile = np.percentile(s0map_data, 2)
            vmax_percentile = np.percentile(s0map_data, 99)

            fig_s0map_epi = plt.figure(figsize=(10, 4), dpi=150)
            plotting.plot_stat_map(
                s0map_nii_path,
                bg_img=None,
                display_mode='mosaic',
                annotate=False,
                cut_coords=None,
                draw_cross=False,
                cmap='gray', # Kept cmap gray for S0 map
                black_bg=False, # Kept black_bg for S0 map
                colorbar=True, # Keep colorbar for maps
                vmin=vmin_percentile,
                vmax=vmax_percentile,
                figure=fig_s0map_epi,
                title=f'S0 map - {task_name}, Run: {run_num}' # Title updated for task/run
            )
            img_buf_s0map_epi = io.BytesIO()
            fig_s0map_epi.savefig(img_buf_s0map_epi, format='png', bbox_inches='tight', pad_inches=0)
            img_buf_s0map_epi.seek(0)
            s0map_montage_base64 = base64.b64encode(img_buf_s0map_epi.read()).decode('utf-8')
            plt.close(fig_s0map_epi)

            with open(html_report_path, "a") as html_report:
                html_report.write(f"<h4>S0 map - {task_name}, Run: {run_num}</h4><img src='data:image/png;base64,{s0map_montage_base64}' alt='S0 map plot_stat_map Montage {task_run_base_name}' style='width: 80%;'>\n") # Alt text updated


            confounds_tsv_file = os.path.join(qc_task_run_session_path, f"{task_run_base_name}_desc-confounds_timeseries.tsv") # Path for task/run
            confounds_df = pd.read_csv(confounds_tsv_file, sep='\t')
            rmse_columns = confounds_df.columns[2:7] # Columns 3 to 7 (0-indexed columns 2 to 6)

            confounds_1d_file = os.path.join(qc_task_run_session_path, f"{task_run_base_name}_desc-confounds_timeseries.1D") # Path for task/run
            confounds_df[rmse_columns].to_csv(confounds_1d_file, sep=' ', header=False, index=False)

            rmse_plot_prefix = os.path.join(qc_task_run_session_path, f"{task_run_base_name}_rmse_plot.png") # Path for task/run


            subprocess.run(plot1d_cmd, check=True)

            rmse_plot_base64 = None # Initialize rmse_plot_base64
            if os.path.exists(rmse_plot_prefix): # Check if the RMSE plot file exists
                try:
                    with open(rmse_plot_prefix, 'rb') as img_file:
                        img_buf = io.BytesIO(img_file.read())
                        rmse_plot_base64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')
                except Exception as embed_e:
                    print(f"      Warning: Could not read/encode generated RMSE plot {rmse_plot_prefix} for HTML embedding: {embed_e}")
                    rmse_plot_base64 = None # Set to None if encoding fails
            else:
                print(f"      Warning: RMSE plot PNG file {rmse_plot_prefix} was not found after 1dplot.py execution.")
                rmse_plot_base64 = None # Set to None if file not found


            with open(html_report_path, "a") as html_report:
                if rmse_plot_base64 is not None: # Only embed if base64 string is valid
                    data_url_rmse = f"data:image/png;base64,{rmse_plot_base64}"
                    html_report.write(f"<h4>RMSE centiles across time for the entire brain for: {task_name}, Run: {run_num}</h4><img src='{data_url_rmse}' alt='RMSE {task_run_base_name}'>\n") # Title/alt updated
                else: # Fallback if embedding fails - just link to the file (or display alt text)
                    html_report.write(f"<h4>RMSE centiles across time for the entire brain for: {task_name}, Run: {run_num}</h4><p>Warning: Could not embed RMSE plot. Check file: {rmse_plot_prefix}</p>\n") # Title updated

                html_report.write("<p>Residual Mean Squared Error (RMSE) indicates the fit quality of the monoexponential T2* decay model, where lower median values for the volume suggest better data quality.</p>\n")

            rmse_prefix = os.path.join(qc_task_run_session_path, f"rmse_{task_run_base_name}") # Path for task/run
            olay_range_98_val = confounds_df[rmse_columns[3]].quantile(0.98) # 98th percentile of the 4th RMSE column (index 3)
            olay_range_98 = f"{olay_range_98_val:.6f}" # Format to string

            # --- Nilearn plot_stat_map montage for RMSE map ---
            rmse_statmap_nii_path = os.path.join(qc_task_run_session_path, f"{task_run_base_name}_desc-rmse_statmap.nii.gz") # Path for task/run
            rmse_statmap_img = nib.load(rmse_statmap_nii_path)
            rmse_statmap_data = rmse_statmap_img.get_fdata()
            vmin_val = 0 # RMSE starts from 0
            vmax_val = float(olay_range_98) # Use 98th percentile as vmax

            fig_rmse_epi = plt.figure(figsize=(10, 4), dpi=150)
            plotting.plot_stat_map(
                rmse_statmap_nii_path,
                bg_img=None,
                display_mode='mosaic',
                annotate=False,
                cut_coords=None,
                draw_cross=False,
                cmap='gray',
                black_bg=True,
                colorbar=True, # Keep colorbar for maps
                vmin=vmin_val,
                vmax=vmax_val,
                figure=fig_rmse_epi,
                title=f'RMSE map - {task_name}, Run: {run_num}' # Title updated for task/run
            )
            img_buf_rmse_epi = io.BytesIO()
            fig_rmse_epi.savefig(img_buf_rmse_epi, format='png', bbox_inches='tight', pad_inches=0)
            img_buf_rmse_epi.seek(0)
            rmse_montage_base64 = base64.b64encode(img_buf_rmse_epi.read()).decode('utf-8')
            plt.close(fig_rmse_epi)


            with open(html_report_path, "a") as html_report:
                html_report.write(f"<h4>RMSE map - {task_name}, Run: {run_num}</h4><img src='data:image/png;base64,{rmse_montage_base64}' alt='RMSE map plot_stat_map Montage {task_run_base_name}'>\n") # Alt text updated


        except subprocess.CalledProcessError as e:
            print(f"      Warning: Multi-echo processing for Task: {task_name}, Run: {run_num} failed at t2smap/related AFNI tools: {e}")
            with open(html_report_path, "a") as html_report:
                html_file.write(f"<p>Multi-echo processing (t2smap, maps) failed for Task: {task_name}, Run: {run_num}.</p>\n")

    return html_report_path

def generate_qc_summary(session_path, qc_session_path, html_report_path):
    """
    Generates Quantitative Summary for DWI and functional MRI data in HTML report
    and saves functional QC measures to a JSON file.
    This function translates the provided bash script module into Python.

    Args:
        session_path (str): Path to the session directory.
        qc_session_path (str): Path to the QC output directory.
        html_report_path (str): Path to the HTML report file.
    """
    import os
    import glob
    import json
    import subprocess
    import numpy as np
    import nibabel as nib
    from nilearn import masking, image
    # from nipype.interfaces import afni  # nipype import is not used in the simplified version, can be removed if not needed in your full script

    def calculate_gsr(epi_file, mask_file, direction):
        """
        Calculates Ghost to Signal Ratio (GSR) in a given direction. Inspired by MRIQC

        Args:
            epi_file (str): Path to the BOLD NIfTI file.
            mask_file (str): Path to the brain mask NIfTI file.
            direction (str): Direction for GSR calculation ('x' or 'y').

        Returns:
            float: GSR value or 'N/A' if calculation fails.
        """
        RAS_AXIS_ORDER = {'x': 0, 'y': 1, 'z': 2, '-x': 0, '-y': 1, '-z': 2}  # Define RAS_AXIS_ORDER

        try:
            epi_img = nib.load(epi_file)
            epi_data = epi_img.get_fdata()
            mask_img = nib.load(mask_file)
            mask_data = mask_img.get_fdata().astype(int)  # Ensure mask is integer type

            direction = direction.lower()
            if direction[-1] not in ('x', 'y', 'all'):
                raise ValueError(f'Unknown direction {direction}, should be one of x, -x, y, -y, all')

            if direction == 'all':
                result = []
                for newdir in ('x', 'y'):
                    result += [calculate_gsr(epi_file, mask_file, newdir)]
                return result  # Not used in current context

            # Roll data of mask through the appropriate axis
            axis = RAS_AXIS_ORDER[direction]
            n2_mask = np.roll(mask_data, mask_data.shape[axis] // 2, axis=axis)

            # Remove from n2_mask pixels inside the brain
            n2_mask = n2_mask * (1 - mask_data)

            # Non-ghost background region is labeled as 2
            n2_mask = n2_mask + 2 * (1 - n2_mask - mask_data)

            # Signal is the entire foreground image
            ghost = np.mean(epi_data[n2_mask == 1]) - np.mean(epi_data[n2_mask == 2])
            signal = np.median(epi_data[n2_mask == 0])

            if signal == 0:  # Avoid division by zero if signal is zero
                return 'N/A'

            return float(ghost / signal)

        except Exception as e:
            print(f"     Warning: GSR calculation failed for direction {direction} on {epi_file}: {e}")
            return 'N/A'

    functional_qc_data_python = {}  # Dictionary to store functional QC metrics for JSON

    with open(html_report_path, 'a') as html_file:  
        html_file.write("<table border='1' style='border-collapse: collapse; width: 100%;'>\n")
        html_file.write("<tr>\n")

        html_file.write(f"<h1>QC Summary Report</h1>\n") 

        # DWI QC Summary - Phase encoding information summary
        html_file.write("<h2>Diffusion MRI: Phase Encoding Information</h2>\n")  # Simplified heading
        html_file.write("<p>NB: Phase encoding polarity extraction requires the presence of PhaseEncodingDirection in the DICOM header, not just the PhaseEncodingAxis.</p>\n")
        html_file.write("<table border='1' style='border-collapse: collapse; width: 50%;'>\n")
        html_file.write("<tr><th>File</th><th>Phase Encoding Direction</th><th>Phase Encoding Axis</th></tr>\n")

        dwi_json_files = glob.glob(os.path.join(session_path, 'dwi', '*.json'))
        dwi_phase_encoding_data = [] # For JSON - Phase encoding info
        for json_file in sorted([f for f in dwi_json_files if 'sbref' not in f]):  # Corrected glob and added sort + sbref filter
            with open(json_file, 'r') as f:
                try:
                    json_data = json.load(f)
                    pe_dir = json_data.get('PhaseEncodingDirection', 'N/A')
                    pe_ax = json_data.get('PhaseEncodingAxis', 'N/A')
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON file: {json_file}")
                    pe_dir = 'N/A'
                    pe_ax = 'N/A'

            if pe_dir == "j": pe_dir_display = "P>>A"
            elif pe_dir == "j-": pe_dir_display = "A>>P"
            elif pe_dir == "i": pe_dir_display = "R>>L"
            elif pe_dir == "i-": pe_dir_display = "L>>R"
            else: pe_dir_display = "N/A"

            if pe_ax == "j" or pe_ax == "j-": pe_ax_display = "COL"
            elif pe_ax == "i" or pe_ax == "i-": pe_ax_display = "ROW"
            else: pe_ax_display = "N/A"

            html_file.write("<tr>\n")
            html_file.write(f"<td>{os.path.basename(json_file).replace('.json', '.nii.gz')}</td><td>{pe_dir_display}</td><td>{pe_ax_display}</td>\n")  # Changed basename to .nii.gz
            html_file.write("</tr>\n")
            dwi_phase_encoding_data.append({  # Store for JSON
                'filename': os.path.basename(json_file).replace('.json', '.nii.gz'),
                'pe_dir': pe_dir_display,
                'pe_ax': pe_ax_display
            })
        html_file.write("</table>\n")

        # DWI QC Summary - Motion/Slice-Drop Corruption Summary
        html_file.write("<h3>Diffusion MRI: Motion/Slice-Drop Corruption Summary</h3>\n")
        html_file.write("<table border='1' style='border-collapse: collapse; width: 80%;'>\n")
        html_file.write("<tr>\n")
        html_file.write("<th>File</th><th>Corrupted Volumes (Count/Total)</th><th>Usable b0 Volumes</th><th>Status</th><th>Volumes Requiring Visual Inspection</th>\n")
        html_file.write("</tr>\n")

        zzbad_files = glob.glob(os.path.join(qc_session_path, '*_dwi_zz_badlist.txt'))
        for zzbad_file in zzbad_files:
            dwi_list_base = os.path.basename(zzbad_file).replace('_dwi_zz_badlist.txt', '')
            dwi_list = f"{dwi_list_base}.nii.gz" # Construct full DWI filename
            dwi_file_path = os.path.join(session_path, 'dwi', dwi_list) # Full path to DWI file

            with open(zzbad_file, 'r') as f_badlist:
                dwi_bad_count = sum(1 for _ in f_badlist) # Efficiently count lines in badlist
                f_badlist.seek(0) # Reset file pointer to read lines again
                dwi_bad_list_str = f_badlist.read().strip() # Read all bad volume indices into a string

            try:
                nv_process = subprocess.run(["3dinfo", "-nv", dwi_file_path], capture_output=True, text=True, check=True)
                nv = int(nv_process.stdout.strip()) # Get total volumes from 3dinfo
            except subprocess.CalledProcessError as e:
                print(f"Warning: 3dinfo failed for {dwi_list}: {e}")
                nv = 'N/A'
            except ValueError:
                nv = 'N/A' # Handle cases where 3dinfo output is not an integer

            xv = 'N/A'
            if isinstance(nv, int):
                xv = nv // 5 # Calculate XV if NV is a valid integer

            usable_b0 = 0
            bval_file = os.path.join(session_path, 'dwi', f"{dwi_list_base}.bval")
            if os.path.exists(bval_file):
                try:
                    with open(bval_file, 'r') as f_bval:
                        bvals = [float(bval) for bval in f_bval.readline().split()] # Read bvals as floats
                        for i, bval in enumerate(bvals):
                            if bval < 10:
                                if not str(i) in dwi_bad_list_str.split(): # Check if volume index is in bad list
                                    usable_b0 += 1
                except Exception as e:
                    print(f"Warning: Error processing bval file {bval_file}: {e}")
                    usable_b0 = 'N/A'
            else:
                usable_b0 = 'N/A'

            status = "Fail" # Default status is Fail
            if isinstance(xv, int) and isinstance(usable_b0, int):
                if dwi_bad_count < xv and usable_b0 >= 1:
                    status = "Pass"

            visual_inspection = "None"
            if dwi_bad_count > 0:
                visual_inspection = dwi_bad_list_str.replace('\n', ', ') # Use comma-separated string for bad volumes

            html_file.write("<tr>\n")
            html_file.write(f"<td>{dwi_list}</td><td>{dwi_bad_count} / {nv}</td><td>{usable_b0}</td><td>{status}</td><td>{visual_inspection}</td>\n")
            html_file.write("</tr>\n")

        html_file.write("</table>\n") 

        # Functional QC Summary - Quantitative Metrics
        html_file.write("<br>\n")  # Added line break for visual separation
        html_file.write("<h2>Resting-state fMRI: Quantitative Metrics</h2>\n") # Simplified heading

        bold_files = sorted(glob.glob(os.path.join(session_path, 'func', '*bold.nii.gz')))  # Corrected glob and added sort
        functional_qc_data_modality = {}  # Store functional QC data for each modality (for JSON)
        for bold_file in bold_files:
            base_name = os.path.basename(bold_file).replace('.nii.gz', '')
            if "_echo-1_" in base_name or "_echo-3_" in base_name:  # Corrected echo- check
                continue

            aor_1d_file = os.path.join(qc_session_path, f"3dToutcount_fraction_{base_name}.1D")
            aqi_1d_file = os.path.join(qc_session_path, f"3dTqual_{base_name}_range.1D")
            fd_1d_file = os.path.join(qc_session_path, f"fwd_{base_name}_abssum.1D")

            try:
                # --- Calculate Mean values directly in Python using NumPy ---
                aor_val_str = np.loadtxt(aor_1d_file)  # Load 1D file into NumPy array
                aqi_val_str = np.loadtxt(aqi_1d_file)
                fd_val_str = np.loadtxt(fd_1d_file)

                aor_val = np.mean(aor_val_str) if aor_val_str.size > 0 else 'N/A'  # Calculate mean using NumPy
                aqi_val = np.mean(aqi_val_str) if aqi_val_str.size > 0 else 'N/A'
                fd_val = np.mean(fd_val_str) if fd_val_str.size > 0 else 'N/A'

            except Exception as e:  # Catch broader exceptions, including nipype errors
                print(f"Warning: Error calculating metric values for {base_name}: {e}")
                aor_val = 'N/A'
                aqi_val = 'N/A'
                fd_val = 'N/A'

            # GSR Calculation
            mean_bold = image.mean_img(bold_file)  # Nilearn mean image calculation
            mask_file = os.path.join(qc_session_path, f"{base_name}_mask.nii.gz")  # Nilearn mask file name
            mask_nifti_img = masking.compute_epi_mask(mean_bold, lower_cutoff=0.2, upper_cutoff=0.9, connected=True)  # Nilearn masking
            mask_nifti_img.to_filename(mask_file)  # Save Nilearn mask
            gsr_x = calculate_gsr(bold_file, mask_file, 'x')  # Calculate GSR_x using new function
            gsr_y = calculate_gsr(bold_file, mask_file, 'y')  # Calculate GSR_y using new function

            functional_qc_data_modality[base_name] = {  # Store per bold file for JSON
                'fd_val': fd_val, 'aor_val': aor_val, 'aqi_val': aqi_val, 'gsr_x': gsr_x, 'gsr_y': gsr_y
            }

            # Generate HTML report for each bold file - Simplified HTML table for quantitative metrics
            html_file.write(f"<h3>{base_name}.nii.gz</h3>\n")
            html_file.write("<table border='1' style='border-collapse: collapse; width: 50%;'>\n")
            html_file.write("<tr><th>Measure</th><th>Value</th></tr>\n")
            html_file.write(f"<tr><td>Mean Framewise Displacement (FD)</td><td>{fd_val:.5f}</td></tr>\n")
            html_file.write(f"<tr><td>Mean Outlier Vox-to-Vol Fraction (AOR)</td><td>{aor_val:.5f}</td></tr>\n")
            html_file.write(f"<tr><td>Mean Distance to Median Volume (AQI)</td><td>{aqi_val:.5f}</td></tr>\n")
            html_file.write(f"<tr><td>Ghost to Signal Ratio (GSR) - X</td><td>{gsr_x:.5f}</td></tr>\n")
            html_file.write(f"<tr><td>Ghost to Signal Ratio (GSR) - Y</td><td>{gsr_y:.5f}</td></tr>\n")
            html_file.write("</table>\n")

            fd_threshold = 0.3
            fd_result_paragraph = "<p>Result: Mean FD Motion within acceptable thresholds.</p>"  # Simplified result message
            if fd_val != 'N/A' and fd_val > fd_threshold:  # Check if fd_val is valid and exceeds threshold
                fd_result_paragraph = "<p>Result: Mean FD Motion EXCEEDS acceptable thresholds (&lt;0.3mm).</p>" # Simplified result message
            html_file.write(f"{fd_result_paragraph}\n")
            html_file.write("<br>\n") # added line break between bold files

        functional_qc_data_python['func'] = functional_qc_data_modality  # Store under 'func' modality

        html_file.write("</table>\n") # Quantitative and Qualitative Summary - Table end

    # Save functional QC data to JSON file
    json_report_path = html_report_path.replace(".html", "_qc_metrics.json") # Create JSON path from HTML path
    with open(json_report_path, 'w') as json_file:
        json.dump(functional_qc_data_python, json_file, indent=4) # Save JSON with indent for readability

    return html_report_path, json_report_path # Return both paths



def generate_3dinfo_summary_table(qc_root, subject_id, html_report_path):
    with open(html_report_path, 'a') as html_file:
        html_file.write("<h2>3dinfo Summary</h2>\n")
        html_file.write("""
        <style>
        table { border-collapse: collapse; width: 80%; }
        th, td { border: 1px solid black; padding: 1px; text-align: left; }
        </style>
        <table>
        <tr>
            <th>Filename</th><th>Mat_x</th><th>Mat_y</th><th>Mat_z</th><th>Vol</th><th>Di</th><th>Dj</th><th>Dk</th><th>TR</th><th>TE</th><th>Orient</th>
        </tr>
        """)

        info_files = glob.glob(os.path.join(qc_root, f"sub-{subject_id}", "**", "*_info.json"), recursive=True) # Recursive glob for JSON

        for info_file_path in sorted(info_files):
            try:
                with open(info_file_path, 'r') as json_file: # Open as JSON file
                    info_dict = json.load(json_file) # Load JSON data into dictionary

                html_file.write("<tr>\n")
                filename = os.path.basename(info_file_path).replace("_info.json", "") # Remove .json extension
                html_file.write(f"<td>{filename}</td>\n")
                html_file.write(f"<td>{info_dict.get('MatrixSize_x', 'N/A')}</td>") # Access from dictionary keys
                html_file.write(f"<td>{info_dict.get('MatrixSize_y', 'N/A')}</td>")
                html_file.write(f"<td>{info_dict.get('MatrixSize_z', 'N/A')}</td>")
                html_file.write(f"<td>{info_dict.get('Volumes', 'N/A')}</td>")
                pd_i = info_dict.get('PixelDimension_i', 'N/A')
                pd_i_str = 'N/A' if pd_i == 'N/A' else f"{float(pd_i):.1f}" if isinstance(pd_i, (int, float)) else pd_i
                html_file.write(f"<td>{pd_i_str}</td>")
                pd_j = info_dict.get('PixelDimension_j', 'N/A')
                pd_j_str = 'N/A' if pd_j == 'N/A' else f"{float(pd_j):.1f}" if isinstance(pd_j, (int, float)) else pd_j
                html_file.write(f"<td>{pd_j_str}</td>")
                st_k = info_dict.get('SliceThickness_k', 'N/A')
                st_k_str = 'N/A' if st_k == 'N/A' else f"{float(st_k):.1f}" if isinstance(st_k, (int, float)) else st_k
                html_file.write(f"<td>{st_k_str}</td>")
                tr_ms = 'N/A' if (tr_val := info_dict.get('TR', 'N/A')) == 'N/A' else f"{(float(tr_val) * 1000):.1f}" if tr_val != 'N/A' else tr_val
                html_file.write(f"<td>{tr_ms}</td>")
                te_ms = 'N/A' if (te_val := info_dict.get('TE', 'N/A')) == 'N/A' else f"{(float(te_val)):.0f}" if te_val != 'N/A' else te_val
                html_file.write(f"<td>{te_ms}</td>")
                html_file.write(f"<td>{info_dict.get('Orientation', 'N/A')}</td>")
                html_file.write("</tr>\n")

            except Exception as e:
                print(f"Warning: Error processing info file {info_file_path}: {e}")
                html_file.write("<tr><td colspan='11'>Error reading info from {}</td></tr>\n".format(os.path.basename(info_file_path))) # colspan='11' to match new columns

        html_file.write("</table>\n")


def mri_rating(session_path, qc_session_path, html_report_path): 
    """
    Generates the final QC decisions summary table in the HTML report.

    Args:
        session_path (str): Path to the session directory in BIDS structure (e.g., <bids_root>/sub-01/ses-test).
        qc_session_path (str): Path to the QC session output directory. (Not directly used in this table generation, but included as per your request/context).
        html_report_path (str): Path to the HTML report file.
    """
    bids_root = os.path.dirname(os.path.dirname(session_path)) 
    session_label = os.path.basename(session_path) 
    subject_id = os.path.basename(os.path.dirname(session_path)).replace("sub-", "") 


    with open(html_report_path, 'a') as html_file:
        # Final Pass/Fail Summary Table
        html_file.write("<h2>Final QC Decisions</h2>\n")
        html_file.write("<style>\n")
        html_file.write("table {\n")
        html_file.write("  border-collapse: collapse;\n")
        html_file.write("  width: 100%;\n")
        html_file.write("}\n")
        html_file.write("th, td {\n")
        html_file.write("  border: 1px solid black;\n")
        html_file.write("  padding: 5px;\n")
        html_file.write("  text-align: center;\n")
        html_file.write("}\n")
        html_file.write("button {\n")
        html_file.write("  font-size: 16px;\n")
        html_file.write("  padding: 10px 20px;\n")
        html_file.write("  margin: 5px;\n")
        html_file.write("  border-radius: 5px;\n")
        html_file.write("  border: 1px solid #ccc;\n")
        html_file.write("  cursor: pointer;\n")
        html_file.write("}\n")
        html_file.write("button:hover {\n")
        html_file.write("  background-color: #f0f0f0;\n")
        html_file.write("}\n")
        html_file.write("button.ok {\n")
        html_file.write("  background-color: lightgreen;\n")
        html_file.write("}\n")
        html_file.write("button.warning {\n")
        html_file.write("  background-color: orange;\n")
        html_file.write("}\n")
        html_file.write("button.reject {\n")
        html_file.write("  background-color: lightcoral;\n")
        html_file.write("}\n")
        html_file.write("textarea {\n")
        html_file.write("  width: 90%;\n")
        html_file.write("  padding: 5px;\n")
        html_file.write("  border-radius: 5px;\n")
        html_file.write("  border: 1px solid #ccc;\n")
        html_file.write("}\n")
        html_file.write(".save-button {\n")
        html_file.write("  font-size: 18px;\n")
        html_file.write("  padding: 15px 30px;\n")
        html_file.write("  background-color: #28a745;\n")
        html_file.write("  color: white;\n")
        html_file.write("  border-radius: 10px;\n")
        html_file.write("  border: none;\n")
        html_file.write("  cursor: pointer;\n")
        html_file.write("}\n")
        html_file.write(".save-button:hover {\n")
        html_file.write("  background-color: #218838;\n")
        html_file.write("}\n")
        html_file.write("</style>\n")

        modalities = ['anat', 'dwi', 'func', 'fmap']
        for modality in modalities:
            html_file.write(f"<h3>{modality}</h3>\n")
            html_file.write(f"<table id='{modality}-summary-table'>\n")
            html_file.write("<tr><th>Filename</th><th>QC Status</th><th>Comments</th></tr>\n")

            modality_path = os.path.join(bids_root, f"sub-{subject_id}", session_label, modality) # corrected session variable
            nii_gz_files = sorted(glob.glob(os.path.join(modality_path, "*.nii.gz")))

            for file_path in nii_gz_files:
                filename = os.path.basename(file_path)
                html_file.write("<tr>\n")
                html_file.write(f"  <td>{filename}</td>\n")
                html_file.write("  <td>\n")
                html_file.write(f"    <button onclick=\"markLevel('{filename}', 2, this)\">OK</button>\n")
                html_file.write(f"    <button onclick=\"markLevel('{filename}', 1, this)\">Borderline/Warning</button>\n")
                html_file.write(f"    <button onclick=\"markLevel('{filename}', 0, this)\">Reject/Problem</button>\n")
                html_file.write(f"    <span id='{filename}_result'>Not Reviewed</span>\n")
                html_file.write("  </td>\n")
                html_file.write(f"  <td><textarea id='{filename}_comment' rows='2' cols='40'></textarea></td>\n")
                html_file.write("</tr>\n")

            # Add the "Set All" buttons for the modality
            html_file.write("<tr>\n")
            html_file.write("  <td colspan='3' style='text-align:center;'>\n")
            html_file.write(f"    <button onclick=\"markAllLevel('{modality}', 2)\">Set All to OK</button>\n")
            html_file.write(f"    <button onclick=\"markAllLevel('{modality}', 1)\">Set All to Borderline/Warning</button>\n")
            html_file.write(f"    <button onclick=\"markAllLevel('{modality}', 0)\">Set All to Reject/Problem</button>\n")
            html_file.write("  </td>\n")
            html_file.write("</tr>\n")

            html_file.write("</table>\n")

        # Add the submit button to save the results
        html_file.write("<button class='save-button' onclick=\"saveReport()\">Submit QC and Save Report</button>\n")

        # JavaScript for handling the QC level buttons and comments
        html_file.write("<script>\n")
        html_file.write("  function markLevel(file, level, button) {\n")
        html_file.write("    var levelText = '';\n")
        html_file.write("    if (level === 0) {\n")
        html_file.write("      levelText = 'Reject/Problem';\n")
        html_file.write("      button.classList.add('reject');\n")
        html_file.write("    } else if (level === 1) {\n")
        html_file.write("      levelText = 'Borderline/Warning';\n")
        html_file.write("      button.classList.add('warning');\n")
        html_file.write("    } else if (level === 2) {\n")
        html_file.write("      levelText = 'OK';\n")
        html_file.write("      button.classList.add('ok');\n")
        html_file.write("    }\n")
        html_file.write("    var buttons = button.parentElement.querySelectorAll('button');\n")
        html_file.write("    buttons.forEach(function(btn) {\n")
        html_file.write("      if (btn !== button) {\n")
        html_file.write("        btn.classList.remove('ok', 'warning', 'reject');\n")
        html_file.write("      }\n")
        html_file.write("    });\n")
        html_file.write("    document.getElementById(file + '_result').innerText = levelText;\n")
        html_file.write("    document.getElementById(file + '_result').setAttribute('data-level', level);\n")
        html_file.write("  }\n")

        html_file.write("  function markAllLevel(modality, level) {\n")
        html_file.write("    var rows = document.querySelectorAll('#' + modality + '-summary-table tr');\n")
        html_file.write("    for (var i = 1; i < rows.length; i++) {\n")
        html_file.write("      var file = rows[i].cells[0].innerText;\n")
        html_file.write("      var buttons = rows[i].cells[1].querySelectorAll('button');\n")
        html_file.write("      buttons.forEach(function(btn) {\n")
        html_file.write("        if (btn.textContent === 'OK' && level === 2) {\n")
        html_file.write("          markLevel(file, level, btn);\n")
        html_file.write("        } else if (btn.textContent === 'Borderline/Warning' && level === 1) {\n")
        html_file.write("          markLevel(file, level, btn);\n")
        html_file.write("        } else if (btn.textContent === 'Reject/Problem' && level === 0) {\n")
        html_file.write("          markLevel(file, level, btn);\n")
        html_file.write("        }\n")
        html_file.write("      });\n")
        html_file.write("    }\n")
        html_file.write("  }\n")

        # Modified saveReport function to save as JSON
        html_file.write("  function saveReport() {\n")
        html_file.write("    var reportContent = {\n")
        html_file.write(f"      'subject_id': '{subject_id}',\n")
        html_file.write(f"      'session': '{session_label}',\n")
        html_file.write("      'modalities': {}\n") # Initialize modalities as an empty object
        html_file.write("    };\n")
        html_file.write("    ['anat', 'dwi', 'func', 'fmap'].forEach(function(modality) {\n")
        html_file.write("      var rows = document.querySelectorAll('#' + modality + '-summary-table tr');\n")
        html_file.write("      reportContent.modalities[modality] = []; // Initialize modality array\n") # Init modality array in JSON
        html_file.write("      for (var i = 1; i < rows.length; i++) {\n")
        html_file.write("        var cells = rows[i].cells;\n")
        html_file.write("        if (cells.length >= 2) { // Ensure there are at least 2 cells\n")
        html_file.write("          var file = cells[0].innerText;\n")
        html_file.write("          var levelSpan = cells[1].querySelector('span');\n")
        html_file.write("          if (levelSpan) {\n")
        html_file.write("            var level = levelSpan.getAttribute('data-level');\n")
        html_file.write("            var comment = document.getElementById(file + '_comment').value;\n")
        html_file.write("            reportContent.modalities[modality].push({\n") # Push data to modality array
        html_file.write("              'filename': file,\n")
        html_file.write("              'qc_level': level,\n")
        html_file.write("              'comment': comment\n")
        html_file.write("            });\n")
        html_file.write("          } else {\n")
        html_file.write("            console.error('Level span not found for file:', file);\n")
        html_file.write("          }\n")
        html_file.write("        } else {\n")
        html_file.write("          console.error('Row structure unexpected:', rows[i]);\n")
        html_file.write("        }\n")
        html_file.write("      }\n")
        html_file.write("    });\n")
        html_file.write("    var jsonString = JSON.stringify(reportContent, null, 2); // Beautify JSON output (indentation = 2)\n") # Convert object to JSON string
        html_file.write("    var element = document.createElement('a');\n")
        html_file.write("    element.setAttribute('href', 'data:application/json;charset=utf-8,' + encodeURIComponent(jsonString)); // Set MIME type to application/json\n")
        html_file.write(f"    element.setAttribute('download', 'QC_Report_sub-{subject_id}_{session_label}.json'); // Download as JSON file\n") # Corrected session variable and file extension
        html_file.write("    document.body.appendChild(element);\n")
        html_file.write("    element.click();\n")
        html_file.write("    document.body.removeChild(element);\n")
        html_file.write("  }\n")
        html_file.write("</script>\n")

def main():
    args = parse_arguments()
    subject_id = args.subject_id
    bids_root = args.bids_root
    session_id = args.session
    qc_root_arg = args.qc_root

    qc_root = create_qc_directories(bids_root, qc_root_arg)
    subject_path = check_subject_directory(bids_root, subject_id)
    sessions_to_process = get_sessions_to_process(subject_path, session_id)
    print(f"Processing subject: {subject_path}") 

    start_time = datetime.datetime.now() # Timer in Python

    for session in sessions_to_process:
        if session: # Check if session is NOT an empty string (i.e., there is a session)
            session_path = os.path.join(subject_path, session)
            qc_session_path = os.path.join(qc_root, f"sub-{subject_id}", session)
            html_report_path = os.path.join(qc_root, f"QC_Report_sub-{subject_id}_{session}.html")
        else: # session is an empty string, process at subject level
            session_path = subject_path # session_path is just the subject_path
            qc_session_path = os.path.join(qc_root, f"sub-{subject_id}") # qc_session_path without session part
            html_report_path = os.path.join(qc_root, f"QC_Report_sub-{subject_id}.html") # html_report_path without session part

        print(f"Processing session: {session if session else 'no session (subject-level)'}") # Indicate if session is present or subject-level

        if os.path.exists(html_report_path):
            overwrite_prompt = input(f"QC report for sub-{subject_id} {session if session else ''} already exists. Overwrite? (no/yes, default: no): ").strip().lower()
            if overwrite_prompt == 'yes' or overwrite_prompt == 'y':
                print(f"Overwriting existing QC report and output for sub-{subject_id} {session if session else ''}...")
                os.remove(html_report_path)
                if os.path.exists(qc_session_path): # Check if folder exists before trying to delete
                    shutil.rmtree(qc_session_path)
            else:
                print(f"Skipping QC report generation for sub-{subject_id} {session if session else ''}...")
                continue # Skip to the next session/subject
        else:
            print(f"Generating QC report for sub-{subject_id} {session if session else ''}...")

                    
        os.makedirs(qc_session_path, exist_ok=True)
        generate_html_report_header(html_report_path, subject_id, session)
        process_anatomical_modality(session_path, qc_session_path, html_report_path)
        process_dwi_modality(session_path, qc_session_path, html_report_path)
        process_functional_modality(session_path, qc_session_path, html_report_path)
        process_mefmri_modality(session_path, qc_session_path, html_report_path)
        generate_3dinfo_summary_table(qc_root, subject_id, html_report_path)
        generate_qc_summary(session_path, qc_session_path, html_report_path)
        mri_rating(session_path, qc_session_path, html_report_path)

        end_time = datetime.datetime.now()
        duration = end_time - start_time
        
        print(f"QC report generation completed in {duration}.") # Print duration
        generate_html_report_footer(html_report_path, duration) # Pass duration here

if __name__ == "__main__":
    main()