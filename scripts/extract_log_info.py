import re
import argparse
import csv
import os


def parse_logs(file_path):
    """
    Parse the log file to extract:
      1. Data loading info (store, cluster, train set shape, val set shape)
      2. Evaluation info (store, loss, MAE)

    Returns:
      data_entries: list of dicts with keys:
        ['store', 'cluster', 'x_train', 'y_train', 'x_val', 'y_val']
      eval_entries: list of dicts with keys:
        ['store', 'loss', 'mae']
    """
    data_entries = []
    eval_entries = []

    # Regex patterns
    load_pattern = re.compile(r"Loading pickle data for store (\d+), cluster (\d+)")
    train_shape_pattern = re.compile(r"x_train shape:\s*(\([^)]+\)),\s*y_train shape:\s*(\([^)]+\))")
    val_shape_pattern = re.compile(r"x_val shape:\s*(\([^)]+\)),\s*y_val shape:\s*(\([^)]+\))")
    eval_pattern = re.compile(r"\[Store (\d+)\]\s+Evaluation\s+-\s+Loss:\s+([\d.]+),\s+MAE:\s+([\d.]+)")

    with open(file_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        load_match = load_pattern.search(line)
        if load_match:
            store = load_match.group(1)
            cluster = load_match.group(2)
            if i + 2 < len(lines):
                train_line = lines[i + 1].strip()
                val_line = lines[i + 2].strip()
                train_match = train_shape_pattern.search(train_line)
                val_match = val_shape_pattern.search(val_line)
                if train_match and val_match:
                    data_entries.append({
                        'store': store,
                        'cluster': cluster,
                        'x_train': train_match.group(1),
                        'y_train': train_match.group(2),
                        'x_val': val_match.group(1),
                        'y_val': val_match.group(2),
                    })
                    i += 3
                    continue
        eval_match = eval_pattern.search(line)
        if eval_match:
            eval_entries.append({
                'store': eval_match.group(1),
                'loss': eval_match.group(2),
                'mae': eval_match.group(3)
            })
        i += 1

    return data_entries, eval_entries


def parse_cluster_rounds(cluster_rounds_str):
    """
    Parse a string specifying evaluations per round.

    If the string contains '=', then it should be in the form "0=30,1=12,2=9" (cluster-specific).
    If the string is a single integer (e.g. "30"), that value is used as the default chunk size for all clusters.
    """
    rounds_dict = {}
    if not cluster_rounds_str:
        return rounds_dict
    if "=" not in cluster_rounds_str:
        try:
            value = int(cluster_rounds_str.strip())
            rounds_dict["default"] = value
        except ValueError:
            pass
    else:
        for part in cluster_rounds_str.split(","):
            part = part.strip()
            if "=" in part:
                cluster_str, num_str = part.split("=")
                try:
                    rounds_dict[int(cluster_str.strip())] = int(num_str.strip())
                except ValueError:
                    pass
    return rounds_dict


def chunk_evaluations_by_cluster(data_entries, eval_entries, cluster_chunk_sizes, max_rounds):
    """
    For each evaluation, determine its cluster (from the data entries) and then chunk the evaluations into rounds.
    For each cluster, evaluations are chunked into groups (the chunk size is determined by:
      - the cluster-specific value if provided (e.g. for cluster 0),
      - otherwise the default chunk size if provided,
      - otherwise all evaluations get round 1).
    Leftovers beyond the last complete chunk are assigned round = -1.
    """
    # Build a map: store -> cluster (int)
    store_to_cluster = {}
    for d in data_entries:
        store_to_cluster[d["store"]] = int(d["cluster"])

    # Group evaluations by cluster.
    cluster_eval_map = {}
    for idx, e in enumerate(eval_entries):
        cluster_id = store_to_cluster.get(e["store"], -1)
        e["_original_index"] = idx
        cluster_eval_map.setdefault(cluster_id, []).append(e)

    # Process each cluster's evaluations.
    for cluster_id, eval_list in cluster_eval_map.items():
        eval_list.sort(key=lambda x: x["_original_index"])
        if cluster_id in cluster_chunk_sizes:
            chunk_size = cluster_chunk_sizes[cluster_id]
        elif "default" in cluster_chunk_sizes:
            chunk_size = cluster_chunk_sizes["default"]
        else:
            # If no chunk size is provided for this cluster, mark all as round 1.
            for e in eval_list:
                e["round"] = 1
            continue

        round_num = 1
        start_idx = 0
        total_evals = len(eval_list)
        while start_idx < total_evals and round_num <= max_rounds:
            end_idx = start_idx + chunk_size
            for e in eval_list[start_idx:end_idx]:
                e["round"] = round_num
            start_idx = end_idx
            round_num += 1

        # Any leftover evaluations beyond max_rounds get round = -1.
        for e in eval_list[start_idx:]:
            e["round"] = -1

    # Flatten the grouped evaluations and restore original order.
    all_evals = []
    for eval_list in cluster_eval_map.values():
        all_evals.extend(eval_list)
    all_evals.sort(key=lambda x: x["_original_index"])
    for e in all_evals:
        e["cluster"] = store_to_cluster.get(e["store"], -1)
        del e["_original_index"]

    return all_evals


def save_to_csv(data_entries, eval_entries, output_dir):
    """
    Save data_entries and eval_entries to CSV files per cluster.
    Files are saved as:
      Cluster<cluster>_data.csv
      Cluster<cluster>_eval.csv
    """
    clusters = set()
    for d in data_entries:
        try:
            clusters.add(int(d["cluster"]))
        except ValueError:
            clusters.add(d["cluster"])
    for e in eval_entries:
        clusters.add(e["cluster"])

    for cluster in clusters:
        cluster_data = [d for d in data_entries if int(d["cluster"]) == cluster]
        cluster_eval = [e for e in eval_entries if e["cluster"] == cluster]
        data_path = os.path.join(output_dir, f"Cluster{cluster}_data.csv")
        eval_path = os.path.join(output_dir, f"Cluster{cluster}_eval.csv")

        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["store", "cluster", "x_train", "y_train", "x_val", "y_val"])
            for d in cluster_data:
                writer.writerow([
                    d["store"], d["cluster"],
                    d["x_train"], d["y_train"],
                    d["x_val"], d["y_val"]
                ])

        with open(eval_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["store", "cluster", "round", "loss", "mae"])
            for e in cluster_eval:
                writer.writerow([e["store"], e["cluster"], e["round"], e["loss"], e["mae"]])

        print(f"[INFO] CSV data for Cluster {cluster} saved to: {data_path}, {eval_path}")


def save_to_txt(data_entries, eval_entries, output_dir):
    """
    Save data_entries and eval_entries to TXT files per cluster.
    Files are saved as:
      Cluster<cluster>_data.txt
      Cluster<cluster>_eval.txt
    """
    clusters = set()
    for d in data_entries:
        try:
            clusters.add(int(d["cluster"]))
        except ValueError:
            clusters.add(d["cluster"])
    for e in eval_entries:
        clusters.add(e["cluster"])

    for cluster in clusters:
        cluster_data = [d for d in data_entries if int(d["cluster"]) == cluster]
        cluster_eval = [e for e in eval_entries if e["cluster"] == cluster]
        data_path = os.path.join(output_dir, f"Cluster{cluster}_data.txt")
        eval_path = os.path.join(output_dir, f"Cluster{cluster}_eval.txt")

        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, 'w') as f:
            for d in cluster_data:
                f.write(f"Loading pickle data for store {d['store']}, cluster {d['cluster']}...\n")
                f.write(f"x_train shape: {d['x_train']}, y_train shape: {d['y_train']}\n")
                f.write(f"x_val shape: {d['x_val']}, y_val shape: {d['y_val']}\n\n")

        with open(eval_path, 'w') as f:
            for e in cluster_eval:
                round_str = "EXTRA/UNUSED" if e["round"] == -1 else f"Round {e['round']}"
                f.write(
                    f"[Store {e['store']} - Cluster {e['cluster']}] "
                    f"{round_str} -> Loss: {e['loss']}, MAE: {e['mae']}\n"
                )

        print(f"[INFO] TXT data for Cluster {cluster} saved to: {data_path}, {eval_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract store data loading information and evaluation metrics from log files, and chunk evaluations into federated learning rounds."
    )
    parser.add_argument("log_file", help="Path to the log file")
    parser.add_argument("--format", choices=["csv", "txt"], default="csv",
                        help="Save output in CSV or TXT format (default: csv)")
    parser.add_argument("--cluster-rounds", default=None,
                        help="Specify evaluations per round. Either a single integer (e.g. '30') to apply for all clusters or a mapping (e.g. '0=30,1=12,2=9').")
    parser.add_argument("--max-rounds", type=int, default=10,
                        help="Maximum number of rounds (default: 10)")

    args = parser.parse_args()

    # Save files in the same directory as the log file.
    output_dir = os.path.dirname(os.path.abspath(args.log_file))

    data_entries, raw_eval_entries = parse_logs(args.log_file)
    cluster_chunk_sizes = parse_cluster_rounds(args.cluster_rounds)

    if cluster_chunk_sizes:
        eval_entries = chunk_evaluations_by_cluster(
            data_entries, raw_eval_entries, cluster_chunk_sizes, args.max_rounds
        )
    else:
        # If no cluster-rounds are specified, assign round=1 to all evaluations.
        store_to_cluster = {d["store"]: int(d["cluster"]) for d in data_entries}
        for e in raw_eval_entries:
            e["cluster"] = store_to_cluster.get(e["store"], -1)
            e["round"] = 1
        eval_entries = raw_eval_entries

    if args.format == "csv":
        save_to_csv(data_entries, eval_entries, output_dir)
    else:
        save_to_txt(data_entries, eval_entries, output_dir)


if __name__ == "__main__":
    main()
