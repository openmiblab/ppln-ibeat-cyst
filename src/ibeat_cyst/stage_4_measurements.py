"""Apply radiomics measurements"""

import os
import logging
from pathlib import Path
from collections import defaultdict
from joblib import Parallel, delayed
from tqdm import tqdm

import numpy as np
import vreg
import pydmr
from miblab import pipe
import numpyradiomics


PIPELINE = 'cyst'
STUDY = {'BL': 'Baseline', 'FU': 'Followup'}
UNITS = numpyradiomics.shape_units(3, 'mm')

def _list_files(folder, recursive=False):
    """Return (paths, names) for all files in a folder.

    paths : list of full file paths (pathlib.Path), sorted
    names : list of file names (str), in the same order
    """
    folder = Path(folder)
    pattern = "**/*" if recursive else "*"
    paths = sorted(p for p in folder.glob(pattern) if p.is_file())
    names = [p.name for p in paths]
    return paths, names

def laterality(mask):
    """Return 'L' or 'R' depending on which side of x=0 the mask's center of
    mass lies, in world (RAS+) coordinates: negative x is left, positive x is right.

    mask : nibabel image (e.g. Nifti1Image) holding a 3D mask; any nonzero
           voxel counts as part of the mask.
    """
    voxels = np.argwhere(np.asanyarray(mask) != 0)
    if len(voxels) == 0:
        raise ValueError("mask is empty")

    com = voxels.mean(axis=0)                   # centre of mass in voxel indices
    x = com[0]
    return 'R' if x < mask.shape[0] / 2 else 'L'


def param(p, idx, side): 
    side = 'left' if side=='L' else 'right'
    return f"kidney_cyst_{side}_{str(idx).zfill(3)}-{p}"

def param_kidney(side, parameter, metric): 
    side = 'left' if side=='L' else 'right'
    return f"kidney_cysts_{side}-{parameter}-{metric}"

def kidney_str(side):
    side = 'left' if side=='L' else 'right'
    return f"kidney_{side}"

def cyst_str(side): 
    side = 'left' if side=='L' else 'right'
    return f"kidney_cyst_{side}"


def cyst_summary(per_cyst_results):
    """
    Summarise cyst measurements per kidney.

    Input keys:  (patient_id, study, side, p, idx) -> value
    Output keys: (patient_id, study, side, p) -> {'mean', 'sdev', 'median', 'iqr', 'n'}
    """
    # Group values over the cyst index
    groups = defaultdict(list)
    for (patient_id, study, side, p, idx), v in per_cyst_results.items():
        groups[(patient_id, study, side, p)].append(v)

    summary = {}
    summary_units = {}
    for key, values in groups.items():
        x = np.asarray(values, dtype=float)
        x = x[~np.isnan(x)]  # ignore missing values
        if x.size == 0:
            continue

        result = numpyradiomics.firstorder_roi(x)

        for k, v in result.items():
            summary[key + (k,)] = v

            p = key[-1]
            intensity_unit = UNITS[p]
            summary_units[(p, k)] = numpyradiomics.firstorder_units(intensity_unit)[k]

    return summary, summary_units


def run(build, logfile):

    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)
    dir_cysts = os.path.join(build, 'cyst', 'stage_2_segmentations')
    
    # for database in ['Controls', 'Patients']:
    for database in ['Patients']:

        logging.info(f"--- Running {database} ---")
        db_cysts = os.path.join(dir_cysts, database) 
        db_output = os.path.join(dir_output, database)
        run_db(db_cysts, db_output)

    
    concatenate(dir_output)   


def run_db(db_cysts, dir_output):

    os.makedirs(dir_output, exist_ok=True)
    files, names = _list_files(db_cysts)

    [run_study(dir_output, names[i], cyst_file) for i, cyst_file in enumerate(files)]

    #Parallel(n_jobs=-1)([delayed(run_study)(dir_output, names[i], cyst_file) for i, cyst_file in enumerate(files)])


def run_study(dir_output, name, cyst_file):

    patient_id, study = name[:8], STUDY[name[9:11]]
    logging.info(f"Running patient {patient_id}, study {study}")

    dmr_file = os.path.join(dir_output, f'{patient_id}_{study}')
    if os.path.exists(f'{dmr_file}.dmr.zip'):
        logging.info(f"-- Results already exist, skipping to the next")
        return

    logging.info(f"-- Reading all cyst masks")
    cyst_vol = vreg.read_nifti(cyst_file).to_right_handed()
    cyst_labels = [int(l) for l in np.unique(cyst_vol.values) if l != 0]

    logging.info(f"-- {len(cyst_labels)} cysts found")

    # Init results
    dmr = {
        'data': {}, 
        'pars': {},
        'columns': ['body part', 'cyst number', 'biomarker', 'metric']
    }
    per_cyst_results = {}
    
    for idx in cyst_labels:
        logging.info(f"-- Measuring cyst {idx}")

        # Binary mask
        mask = (cyst_vol.values==idx).astype(np.float32)
        if np.sum(mask) == 0:
            logging.info(f"---- Zero pixels in cyst {idx}, no output")
            continue

        logging.info(f"---- Building single cyst mask")
        side = laterality(mask)
        try:
            vol = vreg.volume(mask, cyst_vol.affine)
        except Exception as e:
            logging.error("---- Error building mask - skipping")
            continue
        
        logging.info(f"---- Computing shape features")
        try:
            results = numpyradiomics.shape(vol.values.T, vol.spacing[::-1])
        except Exception as e:
            logging.error(f"---- Patient {patient_id} - cyst K_{idx} - error computing shapes: {e}")
            continue

        logging.info("---- Storing results in dictionaries")
        dmr['data'] = dmr['data'] | {param(p, idx, side): [param(p, idx, side), UNITS[p], 'float', cyst_str(side), f'{str(idx).zfill(3)}', p, ''] for p in results.keys()}
        dmr['pars'] = dmr['pars'] | {(patient_id, study, param(p, idx, side)): v for p, v in results.items()}

        per_cyst_results |= {(patient_id, study, side, p, idx): v for p, v in results.items()}

    logging.info("---- Computing per-kidney results")
    per_kidney_results, per_kidney_units = cyst_summary(per_cyst_results)

    logging.info("---- Storing per-kidney results in dictionaries")
    for key, value in per_kidney_results.items():
        patient_id, study, side, parameter, metric = key
        var = param_kidney(side, parameter, metric)
        dmr['data'][var] = [var, per_kidney_units[parameter, metric], 'float', kidney_str(side), '', parameter, metric]
        dmr['pars'][patient_id, study, var] = value  

    pydmr.write(dmr_file, dmr)
    logging.info(f"-- Successfully saved results in {dmr_file}")


def concatenate(dir_output):
    for group in ['Controls', 'Patients']:
        logging.info(f"--- Combining results for {group} ---")

        folder = os.path.join(dir_output, group) 
        folder = Path(folder)
        dmr_files = list(folder.rglob("*.dmr.zip"))
        if dmr_files == []:
            logging.info(f"No files in {folder}. Skipping.")
            continue
        
        dmr_files = [str(f) for f in dmr_files]
        dmr_file = os.path.join(Path(dir_output), f'{group}_results.dmr.zip')
        pydmr.concat(dmr_files, dmr_file)
        logging.info(f"Successfully combined files in {dmr_file}.")

        # Create some derived formats for convenience

        # 1. Long format with additional columns (units, type, description)
        long_format_file = os.path.join(Path(dir_output), f'{group}_results_long.csv')
        pydmr.pars_to_long(dmr_file, long_format_file)
        logging.info(f"Successfully built long format datafile {long_format_file}.")

        # 2. Wide format
        wide_format_file = os.path.join(Path(dir_output), f'{group}_results_wide.csv')
        pydmr.pars_to_wide(dmr_file, wide_format_file)
        logging.info(f"Successfully built wide format datafile {long_format_file}.")



if __name__=='__main__':
    
    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    pipe.run_stage(run, BUILD, PIPELINE, __file__)