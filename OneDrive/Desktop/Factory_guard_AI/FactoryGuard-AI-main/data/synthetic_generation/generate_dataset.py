"""
Synthetic Predictive Maintenance Dataset Generator
===================================================
Creates realistic CNC Machine sensor data with controlled failure patterns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

# ---------------------------------------------------
# ✅ PATH SETUP (CRITICAL FIX)
# ---------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------

class SyntheticMaintenanceDataGenerator:

    def __init__(self, n_machines=500, hours_per_machine=4320, failure_rate=0.025):

        self.n_machines = n_machines
        self.hours_per_machine = hours_per_machine
        self.failure_rate = failure_rate
        self.start_date = datetime(2025, 7, 1)

        self.sensor_baselines = {
            'spindle_temp': 45.0,
            'spindle_vibration': 0.15,
            'spindle_speed': 3000,
            'coolant_temp': 25.0,
            'coolant_pressure': 3.5,
            'hydraulic_pressure': 150.0,
            'motor_current': 12.0,
            'power_consumption': 5.5,
            'tool_vibration': 0.08,
            'tool_wear': 0.0,
            'acoustic_emission': 40.0,
            'cutting_force': 250.0,
            'feed_rate': 500.0,
            'axis_x_position': 0.0,
            'axis_y_position': 0.0,
        }

    def generate_operational_settings(self, n_samples):

        shifts = np.random.choice([1, 2, 3], size=n_samples, p=[0.4, 0.35, 0.25])
        materials = np.random.choice([1, 2, 3], size=n_samples, p=[0.5, 0.35, 0.15])
        workload = np.random.uniform(3, 10, size=n_samples)

        return shifts, materials, workload

    def add_noise(self, signal, noise_level=0.05):

        noise = np.random.normal(0, noise_level * np.std(signal), size=len(signal))
        return signal + noise

    def generate_healthy_machine_data(self, hours):

        timestamps = [self.start_date + timedelta(hours=i) for i in range(hours)]

        data = {
            'timestamp': timestamps,
            'hour_of_operation': list(range(hours)),
        }

        shifts, materials, workload = self.generate_operational_settings(hours)
        data['shift'] = shifts
        data['material_type'] = materials
        data['workload_intensity'] = workload

        for sensor, baseline in self.sensor_baselines.items():

            if sensor == 'tool_wear':
                base_signal = np.linspace(0, baseline + 0.5, hours)
                signal = base_signal + np.random.uniform(-0.02, 0.05, hours)
            else:
                base_signal = np.full(hours, baseline)
                variation = np.random.uniform(-0.1, 0.1, hours) * baseline
                signal = base_signal + variation

            data[sensor] = self.add_noise(signal, noise_level=0.03)

        data['failure'] = np.zeros(hours, dtype=int)
        data['hours_to_failure'] = np.full(hours, -1, dtype=int)

        return pd.DataFrame(data)

    def inject_failure_pattern(self, df, failure_hour):

        degradation_window = 120
        start_degradation = max(0, failure_hour - degradation_window)

        degradation_factors = {
            'spindle_temp': 1.5,
            'spindle_vibration': 3.0,
            'motor_current': 1.3,
            'tool_vibration': 4.0,
            'tool_wear': 2.5,
            'acoustic_emission': 1.8,
            'cutting_force': 1.4,
            'hydraulic_pressure': 0.85,
            'coolant_pressure': 0.9,
        }

        for sensor, factor in degradation_factors.items():

            if sensor in df.columns:

                baseline = df[sensor].iloc[start_degradation]
                degradation_hours = failure_hour - start_degradation

                if degradation_hours > 0:

                    curve = np.linspace(0, 1, degradation_hours) ** 2

                    if factor > 1:
                        degradation = baseline * (factor - 1) * curve
                        df.loc[start_degradation:failure_hour-1, sensor] += degradation
                    else:
                        degradation = baseline * (1 - factor) * curve
                        df.loc[start_degradation:failure_hour-1, sensor] -= degradation

        df.loc[failure_hour:, 'failure'] = 1

        for idx in range(start_degradation, failure_hour):
            df.loc[idx, 'hours_to_failure'] = failure_hour - idx

        return df.iloc[:failure_hour + 1]

    def generate_single_machine_data(self, machine_id):

        will_fail = np.random.random() < self.failure_rate

        if will_fail:
            failure_hour = np.random.randint(
                int(self.hours_per_machine * 0.5),
                int(self.hours_per_machine * 0.9)
            )
            hours = failure_hour + 1
        else:
            hours = self.hours_per_machine
            failure_hour = None

        df = self.generate_healthy_machine_data(hours)

        if will_fail:
            df = self.inject_failure_pattern(df, failure_hour)

        df.insert(0, 'machine_id', machine_id)

        return df

    def generate_complete_dataset(self):

        print("\nGenerating synthetic dataset...")
        print(f"Machines: {self.n_machines}")
        print(f"Hours per machine: {self.hours_per_machine}")
        print(f"Target failure rate: {self.failure_rate:.1%}")
        print("-" * 50)

        all_data = []
        failed_machines = 0

        for machine_id in range(1, self.n_machines + 1):

            df = self.generate_single_machine_data(machine_id)
            all_data.append(df)

            if df['failure'].sum() > 0:
                failed_machines += 1

            if machine_id % 50 == 0:
                print(f"Generated {machine_id}/{self.n_machines} machines...")

        dataset = pd.concat(all_data, ignore_index=True)

        print("-" * 50)
        print("✅ Dataset generation complete!")
        print(f"Total records: {len(dataset):,}")
        print(f"Failed machines: {failed_machines}/{self.n_machines}")
        print(f"Failure rate: {dataset['failure'].mean():.2%}")

        return dataset

    def create_train_test_split(self, df, test_size=0.2):

        unique_machines = df['machine_id'].unique()
        np.random.shuffle(unique_machines)

        split_idx = int(len(unique_machines) * (1 - test_size))

        train_df = df[df['machine_id'].isin(unique_machines[:split_idx])]
        test_df = df[df['machine_id'].isin(unique_machines[split_idx:])]

        print("\nTrain-Test Split:")
        print(f"Train machines: {train_df['machine_id'].nunique()}")
        print(f"Test machines: {test_df['machine_id'].nunique()}")

        return train_df, test_df


def main():

    generator = SyntheticMaintenanceDataGenerator(
        n_machines=500,
        hours_per_machine=4320,
        failure_rate=0.025
    )

    dataset = generator.generate_complete_dataset()
    train_df, test_df = generator.create_train_test_split(dataset)

    print("\nSaving datasets...")

    train_df.to_csv(RAW_DATA_DIR / "train_data.csv", index=False)
    print("✅ Saved: data/raw/train_data.csv")

    test_df.to_csv(RAW_DATA_DIR / "test_data.csv", index=False)
    print("✅ Saved: data/raw/test_data.csv")

    dataset.to_csv(RAW_DATA_DIR / "full_dataset.csv", index=False)
    print("✅ Saved: data/raw/full_dataset.csv")

    print("\n🎉 ALL FILES SAVED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
