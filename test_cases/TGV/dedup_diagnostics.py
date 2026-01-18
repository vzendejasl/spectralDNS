import argparse
import csv
import os


def dedup_diagnostics(input_path, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError("Missing input file: {}".format(input_path))

    with open(input_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError("Empty file: {}".format(input_path))

        rows_by_cycle = {}
        total_rows = 0
        for row in reader:
            if not row or len(row) < 2:
                continue
            try:
                cycle = int(float(row[1]))
            except ValueError:
                continue
            total_rows += 1

            rows_by_cycle[cycle] = row

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for cycle in sorted(rows_by_cycle.keys()):
            writer.writerow(rows_by_cycle[cycle])

    removed = max(0, total_rows - len(rows_by_cycle))
    print("Wrote {} rows to {}".format(len(rows_by_cycle), output_path))
    print("Removed {} duplicate rows".format(removed))


def main():
    parser = argparse.ArgumentParser(
        description="Remove duplicate cycles from diagnostics.csv.")
    parser.add_argument(
        "input",
        help="Input diagnostics file to deduplicate.")
    args = parser.parse_args()

    if args.input.lower().endswith(".csv"):
        output_path = args.input[:-4] + "_processed.csv"
    else:
        output_path = args.input + "_processed.csv"
    dedup_diagnostics(args.input, output_path)


if __name__ == "__main__":
    main()
