import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate per-job NMF CSV files into one ordered table."
    )
    parser.add_argument(
        "--jobs-dir",
        default="jobs",
        help="Directory containing one CSV per SLURM job.",
    )
    parser.add_argument(
        "--output",
        default="nmf_results.csv",
        help="Final aggregated CSV path.",
    )
    return parser.parse_args()


def as_int(row, key):
    try:
        return int(row.get(key, 0))
    except ValueError:
        return 0


def experiment_from_job_name(job_name):
    parts = job_name.split("_")

    if len(parts) <= 3:
        return "unknown"

    return "_".join(parts[3:])


def sort_key(row):
    experiment = row.get("experiment", "")
    experiment_rank = 0 if experiment == "official" else 1

    return (
        experiment_rank,
        as_int(row, "pr"),
        as_int(row, "pc"),
        as_int(row, "threads"),
        experiment,
        row.get("job_name", ""),
    )


def read_rows(jobs_dir):
    rows = []

    for path in sorted(jobs_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)

            for row in reader:
                job_name = path.stem
                row = dict(row)
                row["experiment"] = experiment_from_job_name(job_name)
                row["job_name"] = job_name
                row["source_file"] = path.name
                rows.append(row)

    return rows


def write_rows(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise SystemExit("No CSV rows found to aggregate.")

    base_fields = list(rows[0].keys())
    leading_fields = ["experiment", "job_name", "source_file"]
    fieldnames = leading_fields + [
        field for field in base_fields if field not in leading_fields
    ]

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    jobs_dir = Path(args.jobs_dir)
    output = Path(args.output)

    rows = sorted(read_rows(jobs_dir), key=sort_key)
    write_rows(rows, output)

    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
