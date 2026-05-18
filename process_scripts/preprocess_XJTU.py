# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import os
import json
import h5py
import zipfile
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy import interpolate
from typing import List
from pathlib import Path
from scipy.io import loadmat
from batteryml import BatteryData, CycleData, CyclingProtocol
from batteryml.builders import PREPROCESSORS
from batteryml.preprocess.base import BasePreprocessor
from .time_normalization_utils import normalize_cycle_times


@PREPROCESSORS.register()
class XJTUPreprocessor(BasePreprocessor):
    def process(self, parentdir, **kwargs) -> List[BatteryData]:
        cells = []
        paths = []
        force = kwargs.get('force', False)
        cells_files_path = [
            'Batch-1', 'Batch-2', 'Batch-3',
            'Batch-4', 'Batch-5', 'Batch-6'
        ]
        raw_file = Path(parentdir) / 'Battery Dataset.zip'
        extracted_dir = raw_file.parent / 'Battery Dataset'
        parent_dir = Path(parentdir)
        if all((parent_dir / batch).exists() for batch in cells_files_path):
            dataset_dir = parent_dir
        else:
            dataset_dir = extracted_dir

        # Unzip the raw file when the Zenodo archive layout is used.
        if not dataset_dir.exists():
            with zipfile.ZipFile(raw_file, 'r') as zip_ref:
                pbar = zip_ref.namelist()
                if not self.silent:
                    pbar = tqdm(pbar)
                for file in pbar:
                    if not self.silent:
                        pbar.set_description(f'Unzip XJTU file {file}')
                    zip_ref.extract(file, raw_file.parent)
            dataset_dir = extracted_dir
        else:
            if not self.silent:
                tqdm.write('Skipping XJTU dataset, already exists')

        for files_path in cells_files_path:
            mat_path = dataset_dir / files_path
            if not mat_path.exists():
                continue
            mat_files = os.listdir(mat_path)
            mats = [i for i in mat_files if i.endswith('.mat')]
            for mat in mats:
                cells.append(mat)
                paths.append(mat_path)

        process_batteries_num = 0
        skip_batteries_num = 0
        for path, cell in zip(paths, tqdm(cells, desc='Processing XJTU file')):
            cell = cell.split('.mat')[0]
            cell_name = 'XJTU_' + cell
            # Step1: judge whether to skip the processed file
            if not force:
                whether_to_skip = self.check_processed_file(cell_name)
                if whether_to_skip == True:
                    skip_batteries_num += 1
                    continue

            mat = loadmat(str(path / cell))
            data = mat['data']
            summary = mat['summary']
            cycle_dfs = []
            for cycle in range(1, data.shape[1]+1):
                cycle_data_df = get_one_cycle(data, cycle)
                cycle_data_df['cycle_number'] = cycle
                cycle_dfs.append(cycle_data_df)
            cell_df = pd.concat(cycle_dfs, ignore_index=True)

            # split capacity columns
            cell_df = split_capacity_column(cell_df, cycle_number_column_name='cycle_number', current_column_name='current_A', capacity_column_name='capacity_Ah', nominal_capacity=2.0)

            # Step3: organize the cell data
            battery = organize_cell(cell_df, cell_name, path)
            self.dump_single_file(battery)
            process_batteries_num += 1

            if not self.silent:
                tqdm.write(f'File: {battery.cell_id} dumped to pkl file')

        return process_batteries_num, skip_batteries_num

def organize_cell(timeseries_df, name, path):
    cycle_data = []
    effective_cycle_number = 1
    for cycle_index, df in timeseries_df.groupby('cycle_number'):
        description = str(df['description'].iloc[0])
        # Batch-1/2/3/4/6 start with a low-rate capacity test. Later
        # test-capacity cycles are useful SOH observations and are kept.
        if cycle_index == 1 and '[test capacity]' in description:
            continue
        cycle_data.append(CycleData(
            cycle_number=effective_cycle_number,
            voltage_in_V=df['voltage_V'].tolist(),
            current_in_A=df['current_A'].tolist(),
            temperature_in_C=None,
            discharge_capacity_in_Ah=df['discharge_cap'].tolist(),
            charge_capacity_in_Ah=df['charge_cap'].tolist(),
            time_in_s=list(df['relative_time_min'].values * 60)
        ))
        effective_cycle_number += 1
    # Charge Protocol is constant current
    if 'Batch-1' in str(path):
        charge_rate_in_C = 2.0
        discharge_rate_in_C = 1.0
        soc_interval = [0, 1]
        min_voltage_limit_in_V = 2.5
    elif 'Batch-2' in str(path):
        charge_rate_in_C = 3.0
        discharge_rate_in_C = 1.0
        soc_interval = [0, 1]
        min_voltage_limit_in_V = 2.5
    elif 'Batch-3' in str(path):
        charge_rate_in_C = 2.0
        discharge_rate_in_C = 1.0
        soc_interval = [0, 1]
        min_voltage_limit_in_V = 2.5
    elif 'Batch-4' in str(path):
        charge_rate_in_C = 2.0
        discharge_rate_in_C = 1.0
        soc_interval = [0, 1]
        min_voltage_limit_in_V = 3.0
    elif 'Batch-5' in str(path):
        charge_rate_in_C = 0.5
        discharge_rate_in_C = ''
        soc_interval = [0, 1]
        min_voltage_limit_in_V = 3.0
    elif 'Batch-6' in str(path):
        charge_rate_in_C = 2.0
        discharge_rate_in_C = 0.67
        soc_interval = [0, 1]
        min_voltage_limit_in_V = 2.5

    charge_protocol = [CyclingProtocol(
        rate_in_C=charge_rate_in_C, start_soc=0, end_soc=1.0
    )]
    discharge_protocol = [CyclingProtocol(
        rate_in_C=discharge_rate_in_C, start_soc=1.0, end_soc=1
    )]

    # Normalize time data across all cycles
    cycle_data = normalize_cycle_times(cycle_data, name)

    return BatteryData(
        cell_id=name,
        cycle_data=cycle_data,
        form_factor='cylindrical_18650',
        anode_material='graphite',
        cathode_material='LiNi0.5Co0.2Mn0.3O2',
        discharge_protocol=discharge_protocol,
        charge_protocol=charge_protocol,
        nominal_capacity_in_Ah=2.0,
        min_voltage_limit_in_V=min_voltage_limit_in_V,
        max_voltage_limit_in_V=4.2,
        SOC_interval=soc_interval
    )

def get_value(data, cycle,variable):
    variable_name = ['system_time', 'relative_time_min', 'voltage_V', 'current_A', 'capacity_Ah', 'power_Wh',
                     'temperature_C', 'description']
    if isinstance(variable,str):
        variable = variable_name.index(variable)
    assert cycle <= data.shape[1]
    assert variable <= 7
    value = data[0][cycle-1][variable]
    if variable == 7:
        value = value[0]
    else:
        value = value.reshape(-1)
    return value

def get_one_cycle(data, cycle):
    assert cycle <= data.shape[1]
    cycle_data = pd.DataFrame()
    cycle_data['system_time'] = get_value(data, cycle=cycle,variable='system_time')
    cycle_data['relative_time_min'] = get_value(data, cycle=cycle,variable='relative_time_min')
    cycle_data['voltage_V'] = get_value(data, cycle=cycle,variable='voltage_V')
    cycle_data['current_A'] = get_value(data, cycle=cycle,variable='current_A')
    cycle_data['capacity_Ah'] = get_value(data, cycle=cycle,variable='capacity_Ah')
    cycle_data['power_Wh'] = get_value(data, cycle=cycle,variable='power_Wh')
    cycle_data['temperature_C'] = get_value(data, cycle=cycle,variable='temperature_C')
    cycle_data['description'] = get_value(data, cycle=cycle,variable='description')
    return cycle_data

def split_capacity_column(df, cycle_number_column_name, current_column_name, capacity_column_name, nominal_capacity):
    cycle_number = list(set(df[cycle_number_column_name].values))
    for cycle in cycle_number:
        current_records = df.loc[df[cycle_number_column_name] == cycle, current_column_name].values
        current_c_rate = current_records / nominal_capacity
        capacity_records = df.loc[df[cycle_number_column_name] == cycle, capacity_column_name].values

        # get start and end index for charge period
        cutoff_indices = np.nonzero(current_c_rate >= 0.01)
        charge_indices = cutoff_indices[0]

        # get start and end index for discharge period
        cutoff_indices = np.nonzero(current_c_rate <= -0.01)
        discharge_indices = cutoff_indices[0]

        # get index for rest period
        rest_indices = np.nonzero(np.abs(current_c_rate) < 0.01)

        # set the charge and discharge columns
        # format:
        #   if in charging, the discharge columns will be set into 0.
        #   if in discharging, the charge columns will be set into 0.
        #   if in resting, both charge and discharge columns will be set into 0.
        discharge_capacity_records = capacity_records.copy()
        if len(charge_indices) > 0:
            discharge_capacity_records[charge_indices[0]: charge_indices[-1] + 1] = 0
        discharge_capacity_records[rest_indices] = 0

        charge_capacity_records = capacity_records.copy()
        if len(discharge_indices) > 0:
            charge_capacity_records[discharge_indices[0]: discharge_indices[-1] + 1] = 0
        charge_capacity_records[rest_indices] = 0

        df.loc[df[cycle_number_column_name] == cycle, 'discharge_cap'] = discharge_capacity_records
        df.loc[df[cycle_number_column_name] == cycle, 'charge_cap'] = charge_capacity_records

    return df
