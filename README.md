# Some sparse documentation...

This repo contains experimental and analysis codes for the auditory ripple experiment

A full copy of the direcrory, including codes and data, can be found in `ceph/akrami/Peter/AudioRipple`

## Experimental codes

These experiments are designed to be run in person using Bonsai.  This is because we wrote custom routines to get smooth modulation of the waveforms.  The C-sharp code for this ripple is defined in the `Extensions` folder; `RippleGenerator.cs`.

## Data

We have complete data-sets from 11 participants.  Data for each participant consists of 3 files

1. `summary0.csv` - This file contains the trial-by-trial parameters, including the timings and the stimulus, match and response parameters and is sufficient for most analyses
2. `WavParams0.csv` - This file contains the parameters of every buffer sent to the sound-card, timestamped by Bonsai.  The `CommonTime` column is required to synchronise the data with the trial data, TimeOffset describes the time since the ripple first started
3. `waveform0.bin` - This file contains the binary buffers that were actually sent to the sound-card.  The entire contents of this file can be recosntructed from the `WavParams0.csv` file.

Codes exist in the analysis file to parse all of these files to generate both session level figures (histograms and decision rules), as well as example and average trial trajectories.  If the index is larger than 0 at the end of each file name this indicates the session was restarted.  Use the index with the most number of trials.

The data are not backed up on Github, but they are on the external hardrive in the following path

`D:\Old_Laptop_Final_Backup_18_11_2024\SWC\AudioRipple\Data`

## Analysis codes

All analysis codes are in the `Analysis` folder, and are written in Python.  The key dependency is `Jax`, but these codes are NOT designed to be run with GPU acceleration.

In the `main()` function of `analysis.py` basic does to first
0. Initialise the analysis object - `all_data = total_handler()`
1. Load the data - `all_data.get_all_df()
2. Fit posterior categories for rectification - `all_data.git _posterior()`
3. Plot the rectified histogram - `all_data.alt_hist_results()`

Further examples on how to work with the data are in the `sfn_figs.ipynb` notebook, which shows how to produce session level plots


Other functions exist for plotting the mean of the rows and the columns and for displaying these results

## Project status

Data collection for this project is pretty awful.  Due to the expensive ripple calculation, we could not collect data online.  Furthermore, participants found the experience extremely unpleasant.  The task would need to be made faster and easier in order to collect a meaningful amount of data.  

Additionally, its unclear if a circular mean is really apprpriate for this dataset.  It is bounded, which means it is possible to estimate normalised likelihood correctly, but it is not clear what framework should be used to work with these.

Good luck!
