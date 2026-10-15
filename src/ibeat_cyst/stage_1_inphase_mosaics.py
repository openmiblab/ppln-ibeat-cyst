import os
import sys
import logging

from joblib import delayed, Parallel
import numpy as np
import dbdicom as db
from tqdm import tqdm
from miblab_plot import mosaic_overlay
from miblab import pipe

from ibeat_cyst.utils import data

PIPELINE = 'cyst'


def run(build, logfile):
    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)
    dir_dixons = os.path.join(build, 'dixon', 'stage_5_clean_dixon_data')
    dir_masks = os.path.join(build, 'kidneyvol', 'stage_3_edit')

    record = data.dixon_record(build)

    for database in ['Controls', 'Patients']:
        
        logging.info(f"--- Running {database} ---")
        db_data = os.path.join(dir_dixons, database) 
        db_masks = os.path.join(dir_masks, database) 
        db_mosaics = os.path.join(dir_output, database) 
        run_db(db_data, db_masks, db_mosaics, record) 


def run_db(db_data, db_masks, db_mosaics, record):
    logging.info(f"--- Reading masks ---")
    os.makedirs(db_mosaics, exist_ok=True)
    all_masks = db.series(db_masks)
    logging.info(f"Found {len(all_masks)} masks")

    # Loop over the masks
    [run_patient(db_data, db_mosaics, record, mask) for mask in all_masks]

    # Parallel(n_jobs=-1)(
    #     delayed(run_patient)(db_data, db_mosaics, record, mask) 
    #     for mask in all_masks
    # )

def run_patient(db_data, db_mosaics, record, mask):

    patient_id, study = mask[1], mask[2][0]

    logging.info(f"--- Generating mosaics for patient {patient_id} ---")
    sequence = data.dixon_series_desc(record, patient_id, study)
    series_in = [db_data, patient_id, study, f'{sequence}_in_phase']
    png_file = os.path.join(db_mosaics, f'{patient_id}_{study}_{sequence}.png')

    # Skip if file exists
    if os.path.exists(png_file):
        logging.info(f"Mosaic already exists, skipping to the next")
        return
    
    # Load arrays and build ROIs
    logging.info(f"Reading in-phase images")
    try:
        in_arr = db.volume(series_in, verbose=0).to_right_handed().values
    except:
        logging.exception(f"Error reading in-phase images for {patient_id}, {study}, {sequence}.")
        return

    logging.info(f"Reading masks")
    mask_arr = db.volume(mask, verbose=0).to_right_handed().values
    class_map = {1: "kidney_left", 2: "kidney_right"}
    rois = {roi: (mask_arr==idx).astype(np.int16) for idx, roi in class_map.items()}

    logging.info(f"Building mosaic")
    mosaic_overlay(in_arr, rois, png_file, vmin=0, vmax=np.percentile(in_arr, 90), margin=[16,16,2], opacity=0.1)

    logging.info(f"Successfully built mosaic {png_file}.")
    # logging.exception(f"Error building mosaic for {patient_id}, {study}, {sequence}.")


if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    pipe.run_stage(run, BUILD, PIPELINE, __file__)
