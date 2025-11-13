import logging
import os
from datetime import datetime

import h5py
import numpy as np
from brainsets.core import serialize_fn_map
from brainsets.descriptions import (
    BrainsetDescription,
    DeviceDescription,
    SessionDescription,
    SubjectDescription,
)
from brainsets.taxonomy import RecordingTech, Sex, Species, Task
from experanto.configs import DEFAULT_MODALITY_CONFIG

# Experanto imports
from experanto.experiment import Experiment

# Brainsets & TemporalData imports
# [Correction 1]: Import ArrayDict
from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries


def prepare_experanto_session(
    experanto_root, output_dir, dataset_name="sensorium_data"
):
    """
    Converts an Experanto experiment into a Brainsets HDF5 file for POYO.
    """
    logging.basicConfig(level=logging.INFO)

    # 1. Load Experanto Data
    # ---------------------------------------------------------
    logging.info(f"Loading Experanto: {experanto_root}")

    # [Correction 2]: Set cache_data=True.
    # If False (default), accessing ._data on devices often returns None or incomplete data.
    experiment = Experiment(experanto_root, DEFAULT_MODALITY_CONFIG, cache_data=True)

    # 2. Define Metadata
    # ---------------------------------------------------------
    brainset_desc = BrainsetDescription(
        id=dataset_name,
        origin_version="1.0.0",
        derived_version="1.0.0",
        source="local_experanto",
        description="Converted from Experanto dataset",
    )

    subject_desc = SubjectDescription(
        id="mouse_01",
        species=Species.MUS_MUSCULUS,
        sex=Sex.UNKNOWN,  # Good practice to include if known, or handle defaults
    )

    session_desc = SessionDescription(
        id="session_01",
        recording_date=datetime.now(),
        task=Task.FREE_BEHAVIOR,
    )

    device_desc = DeviceDescription(
        id="neuropixels_probe",
        recording_tech=RecordingTech.NEUROPIXELS_SPIKES,
    )

    # 3. Process Neural Data
    # ---------------------------------------------------------
    logging.info("Processing Spikes...")
    resp_dev = experiment.devices["responses"]

    # Ensure data is loaded
    if not hasattr(resp_dev, "_data") or resp_dev._data is None:
        raise ValueError(
            "Experanto data not loaded. Ensure cache_data=True in Experiment init."
        )

    dense_activity = resp_dev._data  # Shape: (Time, Neurons)

    # [Correction 3]: Robust sampling rate access.
    # Experanto devices might not expose .sampling_rate directly as an attribute.
    # It is safer to grab it from the config used to create the device.
    fs = experiment.modality_config["responses"].get("sampling_rate")
    if fs is None:
        # Fallback: check if the device object happens to store it
        fs = getattr(
            resp_dev, "sampling_rate", 30.0
        )  # Default to 30Hz (common for Ca) or 30000Hz (ephys)

    start_time = resp_dev.start_time
    n_samples, n_neurons = dense_activity.shape
    timestamps_arr = start_time + np.arange(n_samples) / fs

    # Sparsify: Find (time, neuron) indices where activity > 0
    t_indices, unit_indices = np.nonzero(dense_activity > 0)
    spike_times = timestamps_arr[t_indices]

    # Sort spikes by time (Required by temporaldata)
    sort_idx = np.argsort(spike_times)
    spike_times = spike_times[sort_idx]
    unit_indices = unit_indices[sort_idx]

    unit_ids = np.array([f"unit_{i}" for i in range(n_neurons)]).astype("S")

    # [Correction 4]: Wrap Spikes in IrregularTimeSeries
    spikes = IrregularTimeSeries(
        timestamps=spike_times, unit_index=unit_indices, domain="auto"
    )

    # [Correction 5]: Wrap Units in ArrayDict
    units = ArrayDict(id=unit_ids)

    # 4. Process Behavior
    # ---------------------------------------------------------
    behavior_dict = {}

    # Helper to get sampling rate safely for behavior
    def get_fs(dev_name, dev_obj):
        return experiment.modality_config[dev_name].get(
            "sampling_rate", getattr(dev_obj, "sampling_rate", None)
        )

    if "treadmill" in experiment.devices:
        logging.info("Processing Treadmill...")
        tm_dev = experiment.devices["treadmill"]
        tm_data = tm_dev._data
        tm_fs = get_fs("treadmill", tm_dev)
        tm_time = tm_dev.start_time + np.arange(len(tm_data)) / tm_fs

        behavior_dict["treadmill"] = IrregularTimeSeries(
            timestamps=tm_time,
            velocity=tm_data,
            domain="auto",
        )

    if "eye_tracker" in experiment.devices:
        logging.info("Processing Eye Tracker...")
        eye_dev = experiment.devices["eye_tracker"]
        eye_data = eye_dev._data
        eye_fs = get_fs("eye_tracker", eye_dev)
        eye_time = eye_dev.start_time + np.arange(len(eye_data)) / eye_fs

        behavior_dict["eye_tracker"] = IrregularTimeSeries(
            timestamps=eye_time,
            position=eye_data,
            domain="auto",
        )

    # 5. Assemble Data Object
    # ---------------------------------------------------------
    data = Data(
        # [Correction 6]: Pass objects directly, DO NOT use .to_dict()
        brainset=brainset_desc,
        subject=subject_desc,
        session=session_desc,
        device=device_desc,
        # Pass the wrapped objects
        spikes=spikes,
        units=units,
        **behavior_dict,
        domain=Interval(start=timestamps_arr[0], end=timestamps_arr[-1]),
    )

    # 6. Create Train/Test Splits
    # ---------------------------------------------------------
    logging.info("Creating Splits...")
    total_duration = data.domain.end - data.domain.start
    train_end = data.domain.start + (total_duration * 0.8)
    val_end = data.domain.start + (total_duration * 0.9)

    data.set_train_domain(Interval(start=data.domain.start, end=train_end))
    data.set_valid_domain(Interval(start=train_end, end=val_end))
    data.set_test_domain(Interval(start=val_end, end=data.domain.end))

    # 7. Save to HDF5
    # ---------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{dataset_name}_{session_desc.id}.h5")

    logging.info(f"Saving to {file_path}...")
    with h5py.File(file_path, "w") as f:
        data.to_hdf5(f, serialize_fn_map=serialize_fn_map)

    logging.info("Done.")


if __name__ == "__main__":
    # Update these paths
    EXPERANTO_DIR = (
        "/mnt/vast-react/projects/neural_foundation_model/dynamic29513-3-5-Video-full/"
    )
    OUTPUT_DIR = (
        "/mnt/vast-react/projects/neural_foundation_model/torch_brain_data/processed/"
    )
    prepare_experanto_session(EXPERANTO_DIR, OUTPUT_DIR)
