import os
import logging

import numpy as np
import dbdicom as db
from tqdm import tqdm
from miblab_plot import mosaic_overlay
from miblab import pipe

from cyst.utils import data

PIPELINE = 'cyst'


def run(build, logfile):

    logging.info("Stage 1 --- Hello World ---")
    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)
    dir_dixons = os.path.join(build, 'dixon', 'stage_5_clean_dixon_data')
    dir_masks = os.path.join(build, 'kidneyvol', 'stage_3_edit')

    for database in ['Controls', 'Patients']:
        db_data = os.path.join(dir_dixons, database) 
        db_masks = os.path.join(dir_masks, database) 
        db_mosaics = os.path.join(dir_output, database) 
        run_db(db_data, db_masks, db_mosaics)


def run_db(db_data, db_masks, db_mosaics):

    record = data.dixon_record()
    class_map = {1: "kidney_left", 2: "kidney_right"}
    os.makedirs(db_mosaics, exist_ok=True)

    # Loop over the masks
    for mask in tqdm(db.series(db_masks), 'Displaying masks..'):

        # Get the corresponding outphase series
        patient_id, study = mask[1], mask[2][0]
        sequence = data.dixon_series_desc(record, patient_id, study)
        series_in = [db_data, patient_id, study, f'{sequence}_in_phase']
        png_file = os.path.join(db_mosaics, f'{patient_id}_{study}_{sequence}.png')

        # Skip if file exists
        if os.path.exists(png_file):
            continue
        
        try:
            # Load arrays and build ROIs
            in_arr = db.volume(series_in).to_right_handed().values
            mask_arr = db.volume(mask).to_right_handed().values
            rois = {roi: (mask_arr==idx).astype(np.int16) for idx, roi in class_map.items()}

            # Build mosaic and log success
            mosaic_overlay(in_arr, rois, png_file, vmin=0, vmax=np.percentile(in_arr, 90), margin=[16,16,2], opacity=0.1)
            logging.info(f"Success building mosaic for {patient_id}, {study}, {sequence}.")

        except:
            logging.exception(f"Error building mosaic for {patient_id}, {study}, {sequence}.")


if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    pipe.run_script(run, BUILD, PIPELINE)
