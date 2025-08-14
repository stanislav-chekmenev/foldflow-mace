import os
import numpy as np
import pandas as pd

from foldflow.data import utils as du
from metrics import calc_tm_score


def collect_sample_results(results_dir):
    """
    Crawls the results directory structure and collects per-sample self-consistency results.
    Returns a dict: {sample_path: pd.DataFrame of results}
    """

    sample_results = {}
    for length_dir in os.listdir(results_dir):
        length_path = os.path.join(results_dir, length_dir)
        if not os.path.isdir(length_path):
            continue
        for sample_dir in os.listdir(length_path):
            sample_path = os.path.join(length_path, sample_dir, "self_consistency")
            csv_path = os.path.join(sample_path, "sc_results.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                sample_results[(length_dir, sample_path)] = df
    return sample_results


def compute_designability_per_length(sample_results, rmsd_threshold=2.0):
    """
    Computes designability fraction statistics per protein length.

    Args:
        sample_results: dict of {(length_dir, sample_dir): DataFrame}
        rmsd_threshold: threshold for designability (default 2.0 Å)

    Returns:
        designability_fraction: dict {length_dir: fraction_designable}
        designability_stds: dict {length_dir: std_designable}
    """
    designability_fraction = {}

    # Group by length_dir
    length_groups = {}
    for (length_dir, _), df in sample_results.items():
        if length_dir not in length_groups:
            length_groups[length_dir] = []
        length_groups[length_dir].append(df)

    # Compute metrics per length
    for length_dir, dfs in length_groups.items():
        designability_values = []

        for df in dfs:
            sample_is_designable = (df["rmsd"] < rmsd_threshold).any()
            designability_values.append(1.0 if sample_is_designable else 0.0)

        if len(designability_values) > 0:
            designability_fraction[length_dir] = np.mean(designability_values)
        else:
            designability_fraction[length_dir] = 0.0

    return designability_fraction


def compute_novelty_per_length(sample_results, rmsd_threshold=2.0, novelty_threshold=0.5):
    """
    Computes novelty statistics per protein length.

    Args:
        sample_results: dict of {(length_dir, sample_dir): DataFrame}
        rmsd_threshold: threshold for designability (default 2.0 Å)
        novelty_threshold: threshold for novelty (default 0.3)

    Returns:
        novelty_fraction: dict {length_dir: fraction_novel}
    """
    novelty_fraction = {}

    # Group by length_dir
    length_groups = {}
    for (length_dir, _), df in sample_results.items():
        if length_dir not in length_groups:
            length_groups[length_dir] = []
        length_groups[length_dir].append(df)

    # Compute metrics per length
    for length_dir, dfs in length_groups.items():
        novelty_values = []

        for df in dfs:
            sample_is_novel = ((df["rmsd"] < rmsd_threshold) & (df["tm_score"] < novelty_threshold)).any()
            novelty_values.append(1.0 if sample_is_novel else 0.0)

        if len(novelty_values) > 0:
            novelty_fraction[length_dir] = np.mean(novelty_values)
        else:
            novelty_fraction[length_dir] = 0.0

    return novelty_fraction


def compute_diversity_per_length(sample_results, rmsd_threshold=2.0, rmsd_col="rmsd"):
    """
    Computes diversity as average pairwise TM-score of designable generated samples per protein length.

    Args:
        sample_results: dict of {(length_dir, sample_dir): DataFrame}
        rmsd_threshold: threshold for designability (default 2.0 Å)
        rmsd_col: column name for RMSD in each CSV

    Returns:
        diversity_scores: dict {length_dir: average pairwise TM-score of designable samples}
    """
    diversity_scores = {}

    # Group by length_dir
    length_groups = {}
    for (length_dir, sample_path), df in sample_results.items():
        if length_dir not in length_groups:
            length_groups[length_dir] = []
        length_groups[length_dir].append((sample_path, df))

    # Compute diversity per length
    for length_dir, samples in length_groups.items():
        # Collect all designable sample structures for this length
        designable_structures = []

        for sample_path, df in samples:
            # Check if this sample is designable (has at least one structure with RMSD < threshold)
            if (df[rmsd_col] < rmsd_threshold).any():
                # Get all designable structures from this sample
                designable_rows = df[df[rmsd_col] < rmsd_threshold]

                for _, row in designable_rows.iterrows():
                    esmf_sample_path = sample_path
                    esmf_sample_path += f"/esmf/sample_{row.iloc[0]}.pdb"
                    try:
                        # Load PDB and extract coordinates and sequence
                        esmf_feats = du.parse_pdb_feats("folded_sample", esmf_sample_path)
                        designable_structures.append((esmf_feats["bb_positions"], row["sequence"]))
                    except Exception as e:
                        # Skip if structure loading fails
                        continue

        if len(designable_structures) < 2:
            # Need at least 2 structures to compute pairwise diversity
            diversity_scores[length_dir] = None
            continue

        # Compute pairwise TM-scores for this length
        tm_scores = []

        for i in range(len(designable_structures)):
            for j in range(i + 1, len(designable_structures)):
                try:
                    pos_1, seq_1 = designable_structures[i]
                    pos_2, seq_2 = designable_structures[j]

                    # Calculate TM-score between structures i and j
                    _, tm_score = calc_tm_score(pos_1, pos_2, seq_1, seq_2)
                    tm_scores.append(tm_score)
                except Exception as e:
                    # Skip if TM-score calculation fails
                    continue

        if len(tm_scores) > 0:
            diversity_scores[length_dir] = np.mean(tm_scores)
        else:
            diversity_scores[length_dir] = None

    return diversity_scores


def compute_scrmsd_per_length(sample_results):
    """
    Computes scRMSD statistics per protein length.

    Args:
        sample_results: dict of {(length_dir, sample_dir): DataFrame}

    Returns:
        mean_scrmsd: dict {length_dir: mean_best_rmsd}
    """
    mean_scrmsd = {}

    # Group by length_dir
    length_groups = {}
    for (length_dir, _), df in sample_results.items():
        if length_dir not in length_groups:
            length_groups[length_dir] = []
        length_groups[length_dir].append(df)

    # Compute metrics per length
    for length_dir, dfs in length_groups.items():
        best_rmsds = []

        for df in dfs:
            df = df.drop(0)
            min_rmsd = df["rmsd"].min()
            best_rmsds.append(min_rmsd)

        if len(best_rmsds) > 0:
            mean_scrmsd[length_dir] = np.mean(best_rmsds)
        else:
            mean_scrmsd[length_dir] = None

    return mean_scrmsd


def find_best_sample_per_length(sample_results):
    """
    Finds the sample path with the lowest scRMSD value for each protein length.

    Args:
        sample_results: dict of {(length_dir, sample_path): DataFrame}

    Returns:
        best_samples: dict {length_dir: (sample_path, best_rmsd)}
    """
    best_samples = {}

    # Group by length_dir
    length_groups = {}
    for (length_dir, sample_path), df in sample_results.items():
        if length_dir not in length_groups:
            length_groups[length_dir] = []
        length_groups[length_dir].append((sample_path, df))

    # Find best sample per length
    for length_dir, samples in length_groups.items():
        best_rmsd = float("inf")
        best_sample_path = None

        for sample_path, df in samples:
            # Skip the first row (index 0) and find minimum RMSD
            df_filtered = df.drop(0) if len(df) > 1 else df
            if len(df_filtered) > 0:
                min_rmsd = df_filtered["rmsd"].min()
                if min_rmsd < best_rmsd:
                    best_rmsd = min_rmsd
                    best_sample_path = sample_path

        if best_sample_path is not None:
            best_samples[length_dir] = (best_sample_path, best_rmsd)
        else:
            best_samples[length_dir] = (None, None)

    return best_samples


def compute_mean_std(metric: dict):
    means = list(metric.values())
    means = [m for m in means if m is not None]  # Filter out None values
    mean_metric = np.mean(means)
    stderr_metric = np.std(means, ddof=1) / np.sqrt(len(means))
    return mean_metric, stderr_metric


if __name__ == "__main__":
    results_dir = "results/ff2_mace_195k"
    sample_results = collect_sample_results(results_dir)

    metrics = {}
    results = {}
    # Find best sample per length
    best_samples = find_best_sample_per_length(sample_results)

    # Per-length metrics
    metrics["scRMSD"] = compute_scrmsd_per_length(sample_results)
    metrics["Designablity"] = compute_designability_per_length(sample_results)
    metrics["Novelty"] = compute_novelty_per_length(sample_results)
    metrics["Diversity"] = compute_diversity_per_length(sample_results)

    # Compute means and std
    for metric_name, metrics_per_length in metrics.items():
        if metric_name.lower() != "diversity":
            results[metric_name] = compute_mean_std(metrics_per_length)
        else:
            results[metric_name] = (compute_mean_std(metrics_per_length)[0], None)

    # Print metrics
    print(f"The results of the folder {results_dir} have been evaluated:")
    for metric_name, mean_std in results.items():
        mean = mean_std[0]
        std = mean_std[1]
        if mean is not None and std is not None:
            print(f"    {metric_name}: {mean:.3f} ± {std:.3f}")
        elif mean is not None:
            print(f"    {metric_name}: {mean:.3f}")
        else:
            print(f"    {metric_name}: None")
