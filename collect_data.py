from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import serial
from serial import SerialException


def collect_data(
    port: str,
    baud_rate: int,
    duration_seconds: float,
    label: str,
    output_file: Path,
) -> None:
    """
    Read acceleration data from the Arduino over USB serial
    and save valid readings to a CSV file.
    """

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Opening serial port {port} at {baud_rate} baud...")

    try:
        arduino = serial.Serial(
            port=port,
            baudrate=baud_rate,
            timeout=1,
        )
    except SerialException as error:
        raise SystemExit(
            f"Could not open {port}.\n"
            f"Make sure the Arduino is connected and Serial Monitor is closed.\n"
            f"Original error: {error}"
        )

    with arduino:
        # Opening a serial connection often resets the Arduino.
        print("Waiting for Arduino to restart...")
        time.sleep(2)

        arduino.reset_input_buffer()

        start_time = time.perf_counter()
        rows_written = 0
        rejected_lines = 0

        run_id = output_file.stem

        with output_file.open(
            mode="w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    "computer_time_s",
                    "sensor_time_us",
                    "accel_x_g",
                    "accel_y_g",
                    "accel_z_g",
                    "label",
                    "run_id",
                ]
            )

            print(
                f"Collecting '{label}' data for "
                f"{duration_seconds:.1f} seconds..."
            )

            while time.perf_counter() - start_time < duration_seconds:
                raw_bytes = arduino.readline()

                if not raw_bytes:
                    continue

                line = raw_bytes.decode(
                    "utf-8",
                    errors="ignore",
                ).strip()

                parts = line.split(",")

                # Ignore headers, errors and malformed lines.
                if len(parts) != 4:
                    rejected_lines += 1
                    continue

                try:
                    sensor_time_us = int(parts[0])
                    accel_x = float(parts[1])
                    accel_y = float(parts[2])
                    accel_z = float(parts[3])
                except ValueError:
                    rejected_lines += 1
                    continue

                writer.writerow(
                    [
                        time.time(),
                        sensor_time_us,
                        accel_x,
                        accel_y,
                        accel_z,
                        label,
                        run_id,
                    ]
                )

                rows_written += 1

    actual_duration = time.perf_counter() - start_time

    print()
    print("Collection complete.")
    print(f"Saved file: {output_file}")
    print(f"Valid readings: {rows_written}")
    print(f"Rejected lines: {rejected_lines}")
    print(f"Actual duration: {actual_duration:.2f} seconds")

    if actual_duration > 0:
        actual_rate = rows_written / actual_duration
        print(f"Approximate sample rate: {actual_rate:.1f} Hz")

        if actual_rate < 150:
            print(
                "Warning: sample rate is much lower than the "
                "200 Hz target."
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect vibration data from an Arduino IMU."
    )

    parser.add_argument(
        "--port",
        required=True,
        help="Arduino serial port, such as COM3.",
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Serial baud rate. Default: 115200.",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Recording duration in seconds. Default: 10.",
    )

    parser.add_argument(
        "--label",
        required=True,
        help="Condition label, such as stationary or normal.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path.",
    )

    arguments = parser.parse_args()

    if arguments.duration <= 0:
        raise SystemExit("Duration must be greater than zero.")

    collect_data(
        port=arguments.port,
        baud_rate=arguments.baud,
        duration_seconds=arguments.duration,
        label=arguments.label,
        output_file=arguments.output,
    )


if __name__ == "__main__":
    main()