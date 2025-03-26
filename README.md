# Fast MRI Quality Control (QC) Script v0.6

This script automates basic quality control checks of MRI data. It reviews the content in anatomical (anat), diffusion-weighted imaging (dwi), and functional (func) folders of a BIDS formatted MRI dataset, generating a QC report in HTML format (typically within 2-3 minutes).

The goal is to complete this quick version of QC on the same day as MRI acquisition, allowing for timely review and decision-making.

## Usage

```bash
python fastMRIQC.py -p <SubjectID> [-b <BIDS_ROOT>] [-s <SESSION>] [-o <QC_ROOT>] [-h]
```



## Requirements:
  - BIDS-formatted data: The MRI data must be organized in BIDS format. Use tools like "dcm2bids" to convert DICOM to BIDS.
  - Install required python libraries with pip ("pip install -r requirements.txt").


## Modality-Specific QC Overview:

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

## Summary:
 
  - Collects and summarizes key MRI acquisition parameters (e.g., matrix size, slice count, number of dynamics, TR, TE, orientation) into an HTML table.
  - Provides a detailed, session-specific QC report in HTML format, for a visual and quantitative assessment of the MRI data quality.

## Decision Buttons in QC Report:

  - After processing, the script generates a QC report in HTML format, which includes buttons for manual review of each file.
  - The buttons allow the user to categorize each file as:
      1. Reject/Problem
      2. Borderline/Warning
      3. OK
  - Users can also provide comments for each file.
  - A "Submit QC and Save Report" button at the end of the report allows saving the final decision for all files, including their QC status and comments, in a text file.
