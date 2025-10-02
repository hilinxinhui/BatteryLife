# **Data structure description**

You can read the data using the read_data_structure.py script. It demonstrates the basic loading and structure of the standardized data in our processed datasets.

Each file contains the following keys:

- cell_id: the file name of the battery.
- cycle_data: all cycling data of the battery.
  - cycle_number: the cycle number for the cycling data of a cycle.
  - current_in_A: the current value for each data point. Unit is ampere (A).
  - voltage_in_V: the voltage value for each data point. Unit is Volt (V).
  - charge_capacity_in_Ah: the charge capacity value for each data point. Unit is ampere-hour (Ah).
  - discharge_capacity_in_Ah: the discharge capacity value for each data point. Unit is ampere-hour (Ah).
  - time_in_s: the time value for each data point whose unit is second(s).
  - temperature_in_C: the temperature value for each data point. Unit is degree Celsius (C). This information is available only if the raw data provides it.
  - internal_resistance_in_ohm: the internal resistance for each data point. Unit is the ohm (Ω). This information is available only if the raw data provides it. 
- form_factor: the format of the battery.
- anode_material: the anode of the battery.
- cathode_material: the cathode of the battery.
- electrolyte_material: the electrolyte of the battery.
- nominal_capacity_in_Ah: the nominal capacity of the battery.
- depth_of_charge: the percentage of capacity charged relative to the nominal capacity during charging.
- depth_of_discharge: the percentage of capacity discharged relative to the nominal capacity during discharging.
- already_spent_cycles: no record. A redundant column inherited from BatteryML.
- max_voltage_limit_in_V: the designed cutoff voltage for charging.
- min_voltage_limit_in_V: the designed cutoff voltage for discharging.
- max_current_limit_in_A: no record. A redundant column inherited from BatteryML.
- min_current_limit_in_A: no record. A redundant column inherited from BatteryML.
- reference: no record, but can be found in the README.md file for each sub-dataset.
- description: no record, but can be found in the README.md file for each sub-dataset.
- charge_protocol: "multi" for multi-stage charging. Otherwise, a value that indicates the current for the constant-current charge. 
- discharge_protocol: "multi" for multi-stage discharging. Otherwise, a value that indicates the current for the constant-current discharge. 
- SOC_interval: the range of state of charge during the degradation test.