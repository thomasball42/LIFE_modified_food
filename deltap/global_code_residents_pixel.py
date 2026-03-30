import argparse
import math
import os
import shutil
import sys
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union

import geopandas as gpd
import numpy as np
from osgeo import gdal
from yirgacheffe.layers import RasterLayer, ConstantLayer

from aoh import IUCNFormatFilename

GOMPERTZ_A = 2.5
GOMPERTZ_B = -14.5
GOMPERTZ_ALPHA = 1

class Season(Enum):
    RESIDENT = 1
    BREEDING = 2
    NONBREEDING = 3

def gen_gompertz(x: float) -> float:
    return math.exp(-math.exp(GOMPERTZ_A + (GOMPERTZ_B * (x ** GOMPERTZ_ALPHA))))

def numpy_gompertz(x: float) -> float:
    return np.exp(-np.exp(GOMPERTZ_A + (GOMPERTZ_B * (x ** GOMPERTZ_ALPHA))))

def find_layer_path(directory: str, taxon_id: int, season: Season) -> Optional[Path]:
    """Search directory for a .tif matching taxon_id and season using IUCNFormatFilename."""
    for path in Path(directory).glob("**/*.tif"):
        try:
            parts = IUCNFormatFilename.of_filename(path)
        except ValueError:
            continue
        if parts.taxon_id == taxon_id and parts.season == season.name:
            return path
    return None

def open_layer_as_float64(filename: str) -> Union[ConstantLayer, RasterLayer]:
    if filename == "nan":
        return ConstantLayer(0.0)
    layer = RasterLayer.layer_from_file(filename)
    if layer.datatype == gdal.GDT_Float64:
        return layer
    layer64 = RasterLayer.empty_raster_layer_like(layer, datatype=gdal.GDT_Float64)
    layer.save(layer64)
    return layer64

def calc_persistence_value(current_aoh: float, historic_aoh: float, exponent_func) -> float:
    sp_p = exponent_func(current_aoh / historic_aoh)
    sp_p_fix = 1 if sp_p > 1 else sp_p
    return sp_p_fix

def process_delta_p(
    current: Union[ConstantLayer, RasterLayer],
    scenario: Union[ConstantLayer, RasterLayer],
    current_aoh: float,
    historic_aoh: float,
    exponent_func_raster
) -> RasterLayer:
    const_layer = ConstantLayer(current_aoh)
    calc_1 = (const_layer - current) + scenario
    new_aoh = RasterLayer.empty_raster_layer_like(current)
    calc_1.save(new_aoh)

    calc_2 = (new_aoh / historic_aoh).numpy_apply(exponent_func_raster)
    calc_2 = calc_2.numpy_apply(lambda chunk: np.where(chunk > 1, 1, chunk))
    new_p = RasterLayer.empty_raster_layer_like(new_aoh)
    calc_2.save(new_p)

    return new_p

def global_code_residents_pixel_ae(
    species_data_path: str,
    current_aohs_path: str,
    scenario_aohs_path: str,
    historic_aohs_path: str,
    exponent: str,
    output_folder: str,
) -> None:
    os.makedirs(output_folder, exist_ok=True)

    os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"
    try:
        filtered_species_info = gpd.read_file(species_data_path)
    except:  # pylint:disable=W0702
        sys.exit(f"Failed to read {species_data_path}")
    taxid = int(filtered_species_info.id_no.values[0])
    season = Season[filtered_species_info.season.values[0]]

    try:
        exp_val = float(exponent)
        z_exponent_func_float = lambda x: np.float_power(x, exp_val)
        z_exponent_func_raster = lambda x: np.float_power(x, exp_val)
    except ValueError:
        if exponent == "gompertz":
            z_exponent_func_float = gen_gompertz
            z_exponent_func_raster = numpy_gompertz
        else:
            sys.exit(f"unrecognised exponent {exponent}")

    match season:
        case Season.RESIDENT:
            current_path = find_layer_path(current_aohs_path, taxid, Season.RESIDENT)
            if current_path is None:
                print(f"Failed to find current layer for taxon {taxid} RESIDENT in {current_aohs_path}")
                sys.exit()

            historic_path = find_layer_path(historic_aohs_path, taxid, Season.RESIDENT)
            if historic_path is None:
                print(f"Failed to find historic layer for taxon {taxid} RESIDENT in {historic_aohs_path}")
                sys.exit()

            # Reuse the exact filename from the current layer for output
            output_filename = current_path.name

            try:
                current = open_layer_as_float64(str(current_path))
            except FileNotFoundError:
                print(f"Failed to open current layer {current_path}")
                sys.exit()

            scenario_path = find_layer_path(scenario_aohs_path, taxid, Season.RESIDENT)
            try:
                scenario = open_layer_as_float64(str(scenario_path) if scenario_path else "nan")
            except FileNotFoundError:
                scenario = ConstantLayer(0.0)

            try:
                historic_aoh = RasterLayer.layer_from_file(str(historic_path)).sum()
            except FileNotFoundError:
                print(f"Failed to open historic layer {historic_path}")
                sys.exit()

            if historic_aoh == 0.0:
                print(f"Historic AoH for {taxid} is zero, aborting")
                sys.exit()

            layers = [current, scenario]
            union = RasterLayer.find_union(layers)
            for layer in layers:
                try:
                    layer.set_window_for_union(union)
                except ValueError:
                    pass

            current_aoh = current.sum()

            new_p_layer = process_delta_p(current, scenario, current_aoh, historic_aoh, z_exponent_func_raster)
            print(new_p_layer.sum())

            old_persistence = calc_persistence_value(current_aoh, historic_aoh, z_exponent_func_float)
            print(old_persistence)
            calc = new_p_layer - ConstantLayer(old_persistence)

            with TemporaryDirectory() as tmpdir:
                tmpfile = os.path.join(tmpdir, output_filename)
                with RasterLayer.empty_raster_layer_like(new_p_layer, filename=tmpfile) as delta_p:
                    calc.save(delta_p)
                shutil.move(tmpfile, os.path.join(output_folder, output_filename))

        case Season.NONBREEDING:
            current_breeding_path = find_layer_path(current_aohs_path, taxid, Season.BREEDING)
            if current_breeding_path is None:
                print(f"Failed to find current breeding layer for taxon {taxid} in {current_aohs_path}")
                sys.exit()

            current_non_breeding_path = find_layer_path(current_aohs_path, taxid, Season.NONBREEDING)
            if current_non_breeding_path is None:
                print(f"Failed to find current non-breeding layer for taxon {taxid} in {current_aohs_path}")
                sys.exit()

            historic_breeding_path = find_layer_path(historic_aohs_path, taxid, Season.BREEDING)
            if historic_breeding_path is None:
                print(f"Historic AoH for breeding {taxid} not found, aborting")
                sys.exit()

            historic_non_breeding_path = find_layer_path(historic_aohs_path, taxid, Season.NONBREEDING)
            if historic_non_breeding_path is None:
                print(f"Historic AoH for non breeding {taxid} not found, aborting")
                sys.exit()

            # Reuse the exact filename from the current non-breeding layer for output
            output_filename = current_non_breeding_path.name

            try:
                with RasterLayer.layer_from_file(str(historic_breeding_path)) as aoh:
                    historic_aoh_breeding = aoh.sum()
                if historic_aoh_breeding == 0.0:
                    print(f"Historic AoH breeding for {taxid} is zero, aborting")
                    sys.exit()
            except FileNotFoundError:
                print(f"Historic AoH for breeding {taxid} not found, aborting")
                sys.exit()

            try:
                with RasterLayer.layer_from_file(str(historic_non_breeding_path)) as aoh:
                    historic_aoh_non_breeding = aoh.sum()
                if historic_aoh_non_breeding == 0.0:
                    print(f"Historic AoH for non breeding {taxid} is zero, aborting")
                    sys.exit()
            except FileNotFoundError:
                print(f"Historic AoH for non breeding {taxid} not found, aborting")
                sys.exit()

            scenario_breeding_path = find_layer_path(scenario_aohs_path, taxid, Season.BREEDING) if scenario_aohs_path != "nan" else None
            scenario_non_breeding_path = find_layer_path(scenario_aohs_path, taxid, Season.NONBREEDING) if scenario_aohs_path != "nan" else None

            try:
                current_breeding = open_layer_as_float64(str(current_breeding_path))
            except FileNotFoundError:
                print(f"Failed to open current breeding {current_breeding_path}")
                sys.exit()
            try:
                current_non_breeding = open_layer_as_float64(str(current_non_breeding_path))
            except FileNotFoundError:
                print(f"Failed to open current non breeding {current_non_breeding_path}")
                sys.exit()
            try:
                scenario_breeding = open_layer_as_float64(str(scenario_breeding_path) if scenario_breeding_path else "nan")
            except FileNotFoundError:
                scenario_breeding = ConstantLayer(0.0)
            try:
                scenario_non_breeding = open_layer_as_float64(str(scenario_non_breeding_path) if scenario_non_breeding_path else "nan")
            except FileNotFoundError:
                scenario_non_breeding = ConstantLayer(0.0)

            layers = [current_breeding, current_non_breeding, scenario_breeding, scenario_non_breeding]
            union = RasterLayer.find_union(layers)
            for layer in layers:
                try:
                    layer.set_window_for_union(union)
                except ValueError:
                    pass

            current_aoh_breeding = current_breeding.sum()
            persistence_breeding = calc_persistence_value(
                current_aoh_breeding,
                historic_aoh_breeding,
                z_exponent_func_float
            )

            current_aoh_non_breeding = current_non_breeding.sum()
            persistence_non_breeding = calc_persistence_value(
                current_aoh_non_breeding,
                historic_aoh_non_breeding,
                z_exponent_func_float
            )

            old_persistence = (persistence_breeding ** 0.5) * (persistence_non_breeding ** 0.5)

            new_p_breeding = process_delta_p(
                current_breeding,
                scenario_breeding,
                current_aoh_breeding,
                historic_aoh_breeding,
                z_exponent_func_raster
            )
            new_p_non_breeding = process_delta_p(
                current_non_breeding,
                scenario_non_breeding,
                current_aoh_non_breeding,
                historic_aoh_non_breeding,
                z_exponent_func_raster
            )
            new_p_layer = (new_p_breeding ** 0.5) * (new_p_non_breeding ** 0.5)

            delta_p_layer = new_p_layer - ConstantLayer(old_persistence)

            with TemporaryDirectory() as tmpdir:
                tmpfile = os.path.join(tmpdir, output_filename)
                with RasterLayer.empty_raster_layer_like(new_p_breeding, filename=tmpfile) as output:
                    delta_p_layer.save(output)
                shutil.move(tmpfile, os.path.join(output_folder, output_filename))

        case Season.BREEDING:
            pass  # covered by the nonbreeding case
        case _:
            sys.exit(f"Unexpected season for species {taxid}: {season}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--speciesdata',
        type=str,
        help="Single species/seasonality geojson",
        required=True,
        dest="species_data_path"
    )
    parser.add_argument(
        '--current_path',
        type=str,
        required=True,
        dest="current_path",
        help="path to species current AOH hex"
    )
    parser.add_argument(
        '--scenario_path',
        type=str,
        required=True,
        dest="scenario_path",
        help="path to species scenario AOH hex"
    )
    parser.add_argument(
        '--historic_path',
        type=str,
        required=False,
        dest="historic_path",
        help="path to species historic AOH hex"
    )
    parser.add_argument('--output_path',
        type=str,
        required=True,
        dest="output_path",
        help="path to save output csv"
    )
    parser.add_argument('--z', dest='exponent', type=str, default='0.25')
    args = parser.parse_args()

    global_code_residents_pixel_ae(
        args.species_data_path,
        args.current_path,
        args.scenario_path,
        args.historic_path,
        args.exponent,
        args.output_path,
    )

if __name__ == "__main__":
    main()