import os
import logging
from pathlib import Path

import numpy as np
import dbdicom as db
from tqdm import tqdm
from miblab_plot import mosaic_overlay
from miblab import pipe
import vreg

from ibeat_cyst.utils import data

PIPELINE = 'cyst'
STUDY = {'BL': 'Baseline', 'FU': 'Followup'}
CLASS_MAP = {1: "kidney_left", 2: "kidney_right"}


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


def run(build, logfile):

    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)
    dir_dixons = os.path.join(build, 'dixon', 'stage_5_clean_dixon_data')
    dir_masks = os.path.join(build, 'kidneyvol', 'stage_3_edit')
    dir_cysts = os.path.join(build, 'cyst', 'stage_2_segmentations')

    record = data.dixon_record(build)
    
    for database in ['Controls', 'Patients']:
        db_mosaics = os.path.join(dir_output, database) 
        os.makedirs(db_mosaics, exist_ok=True)

        db_cysts = os.path.join(dir_cysts, database) 
        files, names = _list_files(db_cysts)

        # Loop over the masks
        for i, cyst_file in enumerate(files):
            logging.info(f"--- Building mosaic for {cyst_file} ---")
            patient_id, study = names[i][:8], STUDY[names[i][9:11]]
            sequence = data.dixon_series_desc(record, patient_id, study)
            png_file = os.path.join(db_mosaics, f'{patient_id}_{study}_{sequence}.png')
            if os.path.exists(png_file):
                logging.info(f"Mosaic already exists, skipping to the next")
                continue

            for site in ['Bari', 'Bordeaux', 'Exeter', 'Leeds', 'Sheffield', 'Turku']:

                logging.info(f"Looking for in-phase data in {site}.")
                db_dixons = os.path.join(dir_dixons, database, site) 
                series_in = [db_dixons, patient_id, study, f'{sequence}_in_phase']
                if not db.exists(series_in):
                    logging.info(f"In-phase data not in {site}.")
                    continue
                logging.info(f"In-phase data found in {site}.")

                # Get the corresponding masks
                db_masks = os.path.join(dir_masks, database, site) 
                series_masks = [db_masks, patient_id, study, f'kidney_masks']
            
                logging.info(f"Reading in-phase images")
                try:
                    in_arr = db.volume(series_in, verbose=0).to_right_handed().values
                except:
                    logging.exception(f"Error reading in-phase images for {patient_id}, {study}, {sequence}.")
                    return

                logging.info(f"Reading masks")
                mask_arr = db.volume(series_masks, verbose=0).to_right_handed().values
                cyst_arr = vreg.read_nifti(cyst_file).to_right_handed().values

                logging.info(f"Building ROIs...")
                cyst_labels = [l for l in np.unique(cyst_arr) if l != 0]
                rois = {roi: (mask_arr==idx).astype(np.int16) for idx, roi in CLASS_MAP.items()}
                rois |= {f"cyst_{idx}": (cyst_arr==idx).astype(np.int16) for idx in cyst_labels}

                logging.info(f"Building mosaic")
                colors = [[1, 0, 0, 0.4], [0, 1, 0, 0.4]] + len(cyst_labels) * [[0, 0, 1, 0.8]]
                mosaic_overlay(in_arr, rois, png_file, vmin=0, vmax=np.percentile(in_arr, 95), margin=[16,16,2], colors=colors)

                logging.info(f"Successfully built mosaic {png_file}.")
                break

                # except:
                #     logging.exception(f"Error building mosaic for {patient_id}, {study}, {sequence}.")


if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    pipe.run_stage(run, BUILD, PIPELINE, __file__)
