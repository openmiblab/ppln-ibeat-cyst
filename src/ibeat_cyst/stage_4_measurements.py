"""Apply radiomics measurements"""

import os
import logging
from pathlib import Path
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
    return 'R' if x > 0 else 'L'


def param(p, idx, side): 
    return f"{side}K-C{str(idx).zfill(3)}-{p}"

def run(build, logfile):

    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)
    dir_cysts = os.path.join(build, 'cyst', 'stage_2_segmentations')
    
    # for database in ['Controls', 'Patients']:
    for database in ['Patients']:

        logging.info(f"--- Running {database} ---")
        db_cysts = os.path.join(dir_cysts, database) 
        db_output = os.path.join(dir_output, database)
        run_db(db_cysts, db_output)

    logging.info(f"--- Combining results ---")
    concatenate(dir_output)   


def run_db(db_cysts, dir_output):

    os.makedirs(dir_output, exist_ok=True)
    files, names = _list_files(db_cysts)

    # [run_study(dir_output, names[i], cyst_file) for i, cyst_file in enumerate(files)]

    Parallel(n_jobs=-1)([delayed(run_study)(dir_output, names[i], cyst_file) for i, cyst_file in enumerate(files)])


def run_study(dir_output, name, cyst_file):

    patient_id, study = name[:8], STUDY[name[9:11]]

    dmr_file = os.path.join(dir_output, f'{patient_id}_{study}')
    if os.path.exists(f'{dmr_file}.dmr.zip'):
        return
    
    logging.info(f"--- Running patient {patient_id}, study {study} ---")

    # Get mask volume 
    cyst_vol = vreg.read_nifti(cyst_file).to_right_handed()
    cyst_labels = [int(l) for l in np.unique(cyst_vol.values) if l != 0]

    # Init results
    dmr = {'data':{}, 'pars':{}}
    all_study_results = {}
    
    # Loop over the cysts
    for idx in cyst_labels:

        logging.info(f"--- Running patient {patient_id}, study {study}, cyst {idx} ---")

        # Binary mask
        mask = (cyst_vol.values==idx).astype(np.float32)
        if np.sum(mask) == 0:
            continue

        side = laterality(mask)
        vol = vreg.volume(mask, cyst_vol.affine)
        
        # Get skimage features
        try:
            results = numpyradiomics.shape(vol.values.T, vol.spacing[::-1])
        except Exception as e:
            logging.error(f"Patient {patient_id} - cyst K_{idx} - error computing shapes: {e}")
            continue

        dmr['data'] = dmr['data'] | {param(p, idx, side): [param(p, idx, side), UNITS[p], 'float'] for p in results.keys()}
        dmr['pars'] = dmr['pars'] | {(patient_id, study, param(p, idx, side)): v for p, v in results.items()}

        all_study_results |= {(p, idx, side): v for p, v in results.items()}

    # Append parsed biomarkers in the dictionary for convenience
    dmr['columns'] = ['kidney', 'cyst', 'biomarker']
    for p in dmr['data']:
        dmr['data'][p] += p.split('-')

    logging.info(f"--- Saving results for patient {patient_id}, study {study} ---")
    pydmr.write(dmr_file, dmr)


def concatenate(dir_output):

    for group in ['Controls', 'Patients']:
        folder = os.path.join(dir_output, group) 
        folder = Path(folder)
        dmr_files = list(folder.rglob("*.dmr.zip"))
        if dmr_files == []:
            continue
        dmr_files = [str(f) for f in dmr_files]
        dmr_file = os.path.join(Path(dir_output), f'{group}_results.dmr.zip')
        pydmr.concat(dmr_files, dmr_file)

        # Create some derived formats for convenience

        # 1. Long format with additional columns (units, type, description)
        long_format_file = os.path.join(Path(dir_output), f'{group}_results_long.csv')
        pydmr.pars_to_long(dmr_file, long_format_file)

        # 2. Wide format
        wide_format_file = os.path.join(Path(dir_output), f'{group}_results_wide.csv')
        pydmr.pars_to_wide(dmr_file, wide_format_file)



if __name__=='__main__':
    
    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    pipe.run_stage(run, BUILD, PIPELINE, __file__)