import json
from datetime import datetime as dt
from pathlib import Path
from shutil import copyfile


import numpy as np
from aind_data_schema.components.identifiers import Code
from aind_data_schema.core.data_description import DataDescription
from aind_data_schema.core.processing import DataProcess, Processing, ProcessStage
from aind_data_schema_models.process_names import ProcessName
from hdmf.common import VectorData
from hdmf_zarr import NWBZarrIO
from pynwb.ophys import DfOverF, Fluorescence, ImageSegmentation, RoiResponseSeries


def pixel_mask_to_array(mask):
    return np.column_stack([mask["x"], mask["y"], mask["weight"]])


def _filter_rrs(series, soma_indices, rt_region):
    """Return a new RoiResponseSeries keeping only the soma-indexed columns."""
    kwargs = {
        "name": series.name,
        "data": series.data[:, soma_indices],
        "unit": series.unit,
        "rois": rt_region,
    }
    if series.rate is not None:
        kwargs["rate"] = series.rate
        kwargs["starting_time"] = series.starting_time or 0.0
    else:
        kwargs["timestamps"] = series.timestamps[:]
    return RoiResponseSeries(**kwargs)


def filter_nwb_to_soma(session_nwb):
    plane_keys = [key for key in session_nwb.processing if key.startswith("plane")]

    for plane_key in plane_keys:
        proc = session_nwb.processing[plane_key]
        orig_ps = proc["image_segmentation"].plane_segmentations["roi_table"]

        df = orig_ps.to_dataframe()
        soma_mask = df["is_soma"] == True
        soma_indices = np.where(soma_mask)[0]

        # Load pixel masks and column data into memory
        pixel_masks = [
            pixel_mask_to_array(orig_ps["pixel_mask"][i]) for i in soma_indices
        ]
        col_data = {
            col: df.loc[soma_mask, col].tolist()
            for col in orig_ps.colnames
            if col != "pixel_mask"
        }

        # Build new plane segmentation with only soma ROIs
        new_img_seg = ImageSegmentation(name="image_segmentation")
        new_ps = new_img_seg.create_plane_segmentation(
            name="roi_table",
            description=orig_ps.description,
            imaging_plane=orig_ps.imaging_plane,
            columns=[
                VectorData(name=col, description=orig_ps[col].description)
                for col in orig_ps.colnames
                if col != "pixel_mask"
            ],
            colnames=[col for col in orig_ps.colnames if col != "pixel_mask"],
        )
        for i, mask in enumerate(pixel_masks):
            new_ps.add_roi(
                pixel_mask=mask,
                **{col: col_data[col][i] for col in col_data},
            )

        new_rt_region = new_ps.create_roi_table_region(
            region=list(range(len(soma_indices))),
            description="Soma-only ROIs",
        )

        # Replace image_segmentation
        del proc.data_interfaces["image_segmentation"]
        proc.add(new_img_seg)

        # --- Filter direct RoiResponseSeries (e.g., events at top level) ---
        direct_rrs = [
            name
            for name, obj in proc.data_interfaces.items()
            if isinstance(obj, RoiResponseSeries)
        ]
        for name in direct_rrs:
            series = proc[name]
            del proc.data_interfaces[name]
            proc.add(_filter_rrs(series, soma_indices, new_rt_region))

        # --- Filter DfOverF containers (dff traces) ---
        dff_containers = [
            name
            for name, obj in proc.data_interfaces.items()
            if isinstance(obj, DfOverF)
        ]
        for name in dff_containers:
            container = proc[name]
            new_dff = DfOverF(name=name)
            for series in container.roi_response_series.values():
                new_dff.add_roi_response_series(
                    _filter_rrs(series, soma_indices, new_rt_region)
                )
            del proc.data_interfaces[name]
            proc.add(new_dff)

        # --- Filter Fluorescence containers (raw fluorescence / events) ---
        fluor_containers = [
            name
            for name, obj in proc.data_interfaces.items()
            if isinstance(obj, Fluorescence)
        ]
        for name in fluor_containers:
            container = proc[name]
            new_fluor = Fluorescence(name=name)
            for series in container.roi_response_series.values():
                new_fluor.add_roi_response_series(
                    _filter_rrs(series, soma_indices, new_rt_region)
                )
            del proc.data_interfaces[name]
            proc.add(new_fluor)

    return session_nwb


if __name__ == "__main__":
    start_time = dt.now()
    input_dir = Path("/data/")
    output_dir = Path("/results/")
    input_nwb_path = next(input_dir.rglob("*.zarr"))
    data_description = next(input_dir.rglob("*data_description.json"))
    subject = next(input_dir.rglob("*subject.json"))
    procedures = next(input_dir.rglob("*procedures.json"))
    acquisition = next(input_dir.rglob("*acquisition.json"))
    instrument = next(input_dir.rglob("*instrument.json"))
    with NWBZarrIO(input_nwb_path, "r") as io:
        nwbfile = io.read()
        filtered_nwbfile = filter_nwb_to_soma(nwbfile)
        with NWBZarrIO(output_dir / "pophys.nwb.zarr", "w") as export_io:
            export_io.export(src_io=io, nwbfile=filtered_nwbfile)

    data_processes = [
        DataProcess(
            name="Filter NWB values",
            code=Code(
                url="https://codeocean.allenneuraldynamics.org/capsule/6186623/tree"
            ),
            stage=ProcessStage.PROCESSING,
            process_type=ProcessName.OTHER,
            experimenters=["Saskia de Vries", "Arielle Leon"],
            start_date_time=start_time,
            notes="Contains filtered data such that only cells classified as somas are present"
        )
    ]

    processing = Processing(data_processes=data_processes)
    processing.write_standard_file(output_directory=output_dir)
    copyfile(procedures, output_dir / "procedures.json")
    copyfile(subject, output_dir / "subject.json")
    copyfile(data_description, output_dir / "data_descripion.json")
    copyfile(acquisition, output_dir / "acquisition.json")
    copyfile(instrument, output_dir / "instrument.json")

    # with open(data_description) as j:
    #     data_description_data = json.load(j)
    # data_description_data = DataDescription(**data_description_data)
    # derived_data_description = DataDescription.from_raw(
    #     data_description_data, process_name="processed"
    # )
    # output_dir_str = str(output_dir / "data_description.json")
    # derived_data_description.write_standard_file(output_directory=output_dir_str)
