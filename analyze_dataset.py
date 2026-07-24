from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "sensor_time_us",
    "accel_x_g",
    "accel_y_g",
    "accel_z_g",
}

AXES = {
    "X": "accel_x_g",
    "Y": "accel_y_g",
    "Z": "accel_z_g",
}


@dataclass
class QualityThresholds:
    min_sample_rate_hz: float = 150.0
    max_sample_rate_cv_percent: float = 5.0
    clipping_warning_percent: float = 0.5
    clipping_failure_percent: float = 2.0
    maximum_gap_factor: float = 1.75


def load_dataset(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    try:
        dataframe = pd.read_csv(file_path)
    except Exception as error:
        raise RuntimeError(
            f"Could not read the CSV file: {error}"
        ) from error

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Dataset is missing required columns: {missing_text}"
        )

    if len(dataframe) < 10:
        raise ValueError(
            "The dataset contains fewer than 10 samples."
        )

    # Convert required columns to numeric values. Invalid entries become NaN.
    for column in REQUIRED_COLUMNS:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    return dataframe


def calculate_timing_metrics(
    dataframe: pd.DataFrame,
    thresholds: QualityThresholds,
) -> dict[str, float | int]:
    timestamps_us = dataframe["sensor_time_us"].dropna().to_numpy(
        dtype=np.float64
    )

    if len(timestamps_us) < 2:
        raise ValueError(
            "Not enough valid timestamps to calculate sample rate."
        )

    time_differences_us = np.diff(timestamps_us)

    # Remove zero or negative jumps before calculating normal timing statistics.
    valid_differences_us = time_differences_us[
        time_differences_us > 0
    ]

    if len(valid_differences_us) == 0:
        raise ValueError(
            "No valid increasing timestamp intervals were found."
        )

    median_interval_us = float(np.median(valid_differences_us))
    mean_interval_us = float(np.mean(valid_differences_us))

    median_sample_rate_hz = 1_000_000.0 / median_interval_us
    mean_sample_rate_hz = 1_000_000.0 / mean_interval_us

    duration_seconds = (
        timestamps_us[-1] - timestamps_us[0]
    ) / 1_000_000.0

    interval_std_us = float(np.std(valid_differences_us))
    interval_mean_us = float(np.mean(valid_differences_us))

    timing_cv_percent = (
        100.0 * interval_std_us / interval_mean_us
        if interval_mean_us > 0
        else float("inf")
    )

    non_increasing_timestamps = int(
        np.count_nonzero(time_differences_us <= 0)
    )

    gap_threshold_us = (
        median_interval_us * thresholds.maximum_gap_factor
    )

    large_gaps = valid_differences_us[
        valid_differences_us > gap_threshold_us
    ]

    # Estimate missing samples within each large gap.
    estimated_missing_samples = int(
        sum(
            max(
                round(gap_us / median_interval_us) - 1,
                0,
            )
            for gap_us in large_gaps
        )
    )

    return {
        "duration_seconds": duration_seconds,
        "median_interval_us": median_interval_us,
        "mean_interval_us": mean_interval_us,
        "median_sample_rate_hz": median_sample_rate_hz,
        "mean_sample_rate_hz": mean_sample_rate_hz,
        "timing_cv_percent": timing_cv_percent,
        "non_increasing_timestamps": non_increasing_timestamps,
        "large_gap_count": int(len(large_gaps)),
        "estimated_missing_samples": estimated_missing_samples,
    }


def calculate_axis_statistics(
    dataframe: pd.DataFrame,
    clip_limit_g: float,
) -> dict[str, dict[str, float | int]]:
    results: dict[str, dict[str, float | int]] = {}

    # A small tolerance catches values printed near the configured range.
    clipping_threshold = clip_limit_g - 0.0001

    for axis_name, column_name in AXES.items():
        values = dataframe[column_name].to_numpy(dtype=np.float64)
        valid_values = values[np.isfinite(values)]

        if len(valid_values) == 0:
            results[axis_name] = {
                "valid_samples": 0,
                "missing_samples": len(values),
                "mean": float("nan"),
                "std": float("nan"),
                "rms_ac": float("nan"),
                "minimum": float("nan"),
                "maximum": float("nan"),
                "peak_to_peak": float("nan"),
                "positive_clips": 0,
                "negative_clips": 0,
                "total_clips": 0,
                "clip_percent": 0.0,
            }
            continue

        centered_values = valid_values - np.mean(valid_values)

        positive_clips = int(
            np.count_nonzero(valid_values >= clipping_threshold)
        )
        negative_clips = int(
            np.count_nonzero(valid_values <= -clipping_threshold)
        )
        total_clips = positive_clips + negative_clips

        results[axis_name] = {
            "valid_samples": int(len(valid_values)),
            "missing_samples": int(
                len(values) - len(valid_values)
            ),
            "mean": float(np.mean(valid_values)),
            "std": float(np.std(valid_values)),
            "rms_ac": float(
                np.sqrt(np.mean(centered_values ** 2))
            ),
            "minimum": float(np.min(valid_values)),
            "maximum": float(np.max(valid_values)),
            "peak_to_peak": float(np.ptp(valid_values)),
            "positive_clips": positive_clips,
            "negative_clips": negative_clips,
            "total_clips": total_clips,
            "clip_percent": (
                100.0 * total_clips / len(valid_values)
            ),
        }

    return results


def determine_quality_status(
    dataframe: pd.DataFrame,
    timing: dict[str, float | int],
    axis_stats: dict[str, dict[str, float | int]],
    thresholds: QualityThresholds,
) -> tuple[str, list[str], list[str]]:
    warnings: list[str] = []
    failures: list[str] = []

    sample_rate = float(timing["median_sample_rate_hz"])
    timing_cv = float(timing["timing_cv_percent"])
    missing_samples = int(timing["estimated_missing_samples"])
    non_increasing = int(timing["non_increasing_timestamps"])

    if sample_rate < thresholds.min_sample_rate_hz:
        failures.append(
            f"Sample rate is only {sample_rate:.1f} Hz."
        )

    if timing_cv > thresholds.max_sample_rate_cv_percent:
        warnings.append(
            "Timestamp spacing is inconsistent "
            f"(CV = {timing_cv:.2f}%)."
        )

    if missing_samples > 0:
        warnings.append(
            f"Approximately {missing_samples} sample(s) may be missing."
        )

    if non_increasing > 0:
        failures.append(
            f"{non_increasing} timestamp(s) did not increase."
        )

    invalid_rows = int(
        dataframe[
            list(REQUIRED_COLUMNS)
        ].isna().any(axis=1).sum()
    )

    if invalid_rows > 0:
        warnings.append(
            f"{invalid_rows} row(s) contain missing or invalid values."
        )

    for axis_name, stats in axis_stats.items():
        clip_percent = float(stats["clip_percent"])
        standard_deviation = float(stats["std"])

        if clip_percent > thresholds.clipping_failure_percent:
            failures.append(
                f"{axis_name}-axis clipping is "
                f"{clip_percent:.2f}%."
            )
        elif clip_percent > thresholds.clipping_warning_percent:
            warnings.append(
                f"{axis_name}-axis clipping is "
                f"{clip_percent:.2f}%."
            )

        if np.isfinite(standard_deviation) and standard_deviation < 0.0001:
            warnings.append(
                f"{axis_name}-axis is nearly constant."
            )

    if failures:
        status = "FAIL"
    elif warnings:
        status = "WARNING"
    else:
        status = "PASS"

    return status, warnings, failures


def create_time_domain_plot(
    dataframe: pd.DataFrame,
    file_stem: str,
    output_directory: Path,
) -> Path:
    valid_time = dataframe["sensor_time_us"].notna()

    plotting_data = dataframe.loc[valid_time].copy()

    start_time_us = plotting_data["sensor_time_us"].iloc[0]
    plotting_data["time_s"] = (
        plotting_data["sensor_time_us"] - start_time_us
    ) / 1_000_000.0

    output_path = output_directory / f"{file_stem}_time_domain.png"

    plt.figure(figsize=(12, 6))

    for axis_name, column_name in AXES.items():
        plt.plot(
            plotting_data["time_s"],
            plotting_data[column_name],
            label=f"{axis_name} axis",
            linewidth=0.8,
        )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Acceleration (g)")
    plt.title(f"Acceleration vs. Time — {file_stem}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def create_fft_plot(
    dataframe: pd.DataFrame,
    sample_rate_hz: float,
    file_stem: str,
    output_directory: Path,
) -> Path:
    output_path = output_directory / f"{file_stem}_fft.png"

    plt.figure(figsize=(12, 6))

    for axis_name, column_name in AXES.items():
        values = dataframe[column_name].dropna().to_numpy(
            dtype=np.float64
        )

        if len(values) < 2:
            continue

        # Remove static gravity and sensor offset.
        centered = values - np.mean(values)

        # Apply a Hann window to reduce spectral leakage.
        window = np.hanning(len(centered))
        windowed_signal = centered * window

        frequencies = np.fft.rfftfreq(
            len(windowed_signal),
            d=1.0 / sample_rate_hz,
        )

        fft_values = np.fft.rfft(windowed_signal)

        # Amplitude normalization for a one-sided spectrum.
        amplitude = (
            2.0 * np.abs(fft_values) / np.sum(window)
        )

        plt.plot(
            frequencies,
            amplitude,
            label=f"{axis_name} axis",
            linewidth=0.9,
        )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Acceleration amplitude (g)")
    plt.title(f"FFT Spectrum — {file_stem}")
    plt.xlim(0, sample_rate_hz / 2)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def build_report(
    file_path: Path,
    dataframe: pd.DataFrame,
    timing: dict[str, float | int],
    axis_stats: dict[str, dict[str, float | int]],
    status: str,
    warnings: list[str],
    failures: list[str],
    clip_limit_g: float,
    plot_paths: list[Path],
) -> str:
    lines: list[str] = []

    lines.append("=" * 64)
    lines.append("VIBRATION DATASET QUALITY REPORT")
    lines.append("=" * 64)
    lines.append(f"File: {file_path}")
    lines.append(f"Overall status: {status}")
    lines.append("")

    lines.append("DATASET")
    lines.append("-" * 64)
    lines.append(f"Rows: {len(dataframe)}")
    lines.append(
        f"Duration: {float(timing['duration_seconds']):.3f} s"
    )
    lines.append(
        "Median sample rate: "
        f"{float(timing['median_sample_rate_hz']):.2f} Hz"
    )
    lines.append(
        "Mean sample rate: "
        f"{float(timing['mean_sample_rate_hz']):.2f} Hz"
    )
    lines.append(
        "Median timestamp interval: "
        f"{float(timing['median_interval_us']):.2f} µs"
    )
    lines.append(
        "Timing variation: "
        f"{float(timing['timing_cv_percent']):.2f}% CV"
    )
    lines.append(
        f"Large timestamp gaps: {timing['large_gap_count']}"
    )
    lines.append(
        "Estimated missing samples: "
        f"{timing['estimated_missing_samples']}"
    )
    lines.append(
        "Non-increasing timestamps: "
        f"{timing['non_increasing_timestamps']}"
    )
    lines.append("")

    lines.append("ACCELERATION STATISTICS")
    lines.append("-" * 64)

    for axis_name, stats in axis_stats.items():
        lines.append(f"{axis_name} axis")
        lines.append(
            f"  Valid samples: {stats['valid_samples']}"
        )
        lines.append(
            f"  Missing samples: {stats['missing_samples']}"
        )
        lines.append(f"  Mean: {float(stats['mean']):.5f} g")
        lines.append(f"  Std: {float(stats['std']):.5f} g")
        lines.append(
            f"  AC RMS: {float(stats['rms_ac']):.5f} g"
        )
        lines.append(
            f"  Minimum: {float(stats['minimum']):.5f} g"
        )
        lines.append(
            f"  Maximum: {float(stats['maximum']):.5f} g"
        )
        lines.append(
            "  Peak-to-peak: "
            f"{float(stats['peak_to_peak']):.5f} g"
        )
        lines.append("")

    lines.append("CLIPPING / SATURATION")
    lines.append("-" * 64)
    lines.append(
        f"Configured accelerometer limit: ±{clip_limit_g:g} g"
    )

    for axis_name, stats in axis_stats.items():
        lines.append(
            f"{axis_name}: "
            f"+limit={stats['positive_clips']}, "
            f"-limit={stats['negative_clips']}, "
            f"total={stats['total_clips']} "
            f"({float(stats['clip_percent']):.3f}%)"
        )

    lines.append("")
    lines.append("QUALITY FINDINGS")
    lines.append("-" * 64)

    if not warnings and not failures:
        lines.append("No quality problems detected.")

    for failure in failures:
        lines.append(f"FAIL: {failure}")

    for warning in warnings:
        lines.append(f"WARNING: {warning}")

    lines.append("")
    lines.append("GENERATED FILES")
    lines.append("-" * 64)

    for plot_path in plot_paths:
        lines.append(str(plot_path))

    lines.append("")
    lines.append(
        "Note: rejected serial lines cannot be recovered from the CSV. "
        "That count must be recorded by collect_data.py during acquisition."
    )
    lines.append("=" * 64)

    return "\n".join(lines)


def analyze_dataset(
    file_path: Path,
    clip_limit_g: float,
    output_directory: Path | None,
) -> None:
    thresholds = QualityThresholds()
    dataframe = load_dataset(file_path)

    if output_directory is None:
        output_directory = file_path.parent / "analysis"

    output_directory.mkdir(parents=True, exist_ok=True)

    timing = calculate_timing_metrics(dataframe, thresholds)
    axis_stats = calculate_axis_statistics(
        dataframe,
        clip_limit_g,
    )

    status, warnings, failures = determine_quality_status(
        dataframe,
        timing,
        axis_stats,
        thresholds,
    )

    time_plot = create_time_domain_plot(
        dataframe,
        file_path.stem,
        output_directory,
    )

    fft_plot = create_fft_plot(
        dataframe,
        float(timing["median_sample_rate_hz"]),
        file_path.stem,
        output_directory,
    )

    report = build_report(
        file_path=file_path,
        dataframe=dataframe,
        timing=timing,
        axis_stats=axis_stats,
        status=status,
        warnings=warnings,
        failures=failures,
        clip_limit_g=clip_limit_g,
        plot_paths=[time_plot, fft_plot],
    )

    report_path = (
        output_directory / f"{file_path.stem}_quality_report.txt"
    )

    report_path.write_text(report, encoding="utf-8")

    print(report)
    print()
    print(f"Report saved to: {report_path}")
    print(f"Time plot saved to: {time_plot}")
    print(f"FFT plot saved to: {fft_plot}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the quality of an MPU6500 vibration CSV dataset."
        )
    )

    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to the CSV dataset.",
    )

    parser.add_argument(
        "--clip-limit",
        type=float,
        default=8.0,
        help=(
            "Configured accelerometer range in g. "
            "Example: 8 for a ±8 g range. Default: 8."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for plots and the report. "
            "Default: an analysis folder beside the CSV."
        ),
    )

    arguments = parser.parse_args()

    if arguments.clip_limit <= 0:
        raise SystemExit("--clip-limit must be greater than zero.")

    try:
        analyze_dataset(
            file_path=arguments.dataset,
            clip_limit_g=arguments.clip_limit,
            output_directory=arguments.output_dir,
        )
    except Exception as error:
        raise SystemExit(f"Analysis failed: {error}") from error


if __name__ == "__main__":
    main()