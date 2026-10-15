# Cyst quantification in iBEAt

## Installation

Requirements: python (and miniconda, if using conda)

Naviate to the folder and set up the conda environment:

```bash
conda env create -f env.yml
```

Then activate

```bash
conda activate ibeat_cyst
```

For convenience, define a data variable to point to the iBEAt_Build database, so you don't have to type the full path every time. Make sure to change this path so it points to your location of the data:

```bash
$data = "C:\Users\MyUserName\Documents\Data\iBEAt_Build"
```

Run the first stage to generate mosaics for spotting cysts

```bash
python -m ibeat_cyst.stage_1_inphase_mosaics --build=$data
```

Once you have identified the subjects that have cysts, segment the cysts 
manually and save the results in the folder *stage_2_segmentations*. After that you can run the next stage to create mosaics for checking your segmentations:

```bash
python -m ibeat_cyst.stage_3_segmentation_mosaics --build=$data
```

If you see any issues, open the segmentation files up again and edit the cyst masks. When you are happy with all cysts, run the measurements:

```bash
python -m ibeat_cyst.stage_4_measurements --build=$data
```

