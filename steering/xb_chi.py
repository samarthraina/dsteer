import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
# Ensure unsloth is installed: pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# Or other variants depending on your system
# Make unsloth import optional (only needed for load_model_and_tokenizer function)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Warning: transformers not found. Assuming Unsloth handles model/tokenizer loading.")
    AutoModelForCausalLM, AutoTokenizer = None, None # Placeholder
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
# Ensure datasets is installed: pip install datasets
from datasets import load_dataset
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
import random
# Consider installing umap if you want to use it later: pip install umap-learn
try:
    import umap
except ImportError:
    print("Info: umap-learn not installed. UMAP dimensionality reduction unavailable.")
    umap = None
import os
from sklearn.decomposition import PCA
from collections import defaultdict
import argparse # Added for CLI arguments
import time # For timestamping outputs


def xie_beni_index(X, centers, membership, m=2):
    """
    Compute the Xie–Beni index for fuzzy clustering.

    Parameters:
        X : np.ndarray
            Data points with shape (n, d) where n is the number of points and d is the dimension.
        centers : np.ndarray
            Cluster centers with shape (c, d) where c is the number of clusters.
        membership : np.ndarray
            Membership matrix with shape (c, n) where membership[i, k] is the membership degree
            of data point k in cluster i. Can be hard (0/1) or fuzzy.
        m : float, optional (default=2)
            Fuzziness exponent. Typical values are greater than 1.

    Returns:
        float
            The Xie–Beni index. A lower value indicates a better partition.
            Returns np.inf if the minimum center separation is zero or centers cannot be computed.
    """
    n = X.shape[0]     # number of data points
    c = centers.shape[0]  # number of clusters

    if c < 2:
        #print("Warning: Cannot compute Xie-Beni index with fewer than 2 clusters.")
        return np.inf # Or some other indicator of inability to compute

    # Calculate the numerator: weighted sum of squared distances from data points to their cluster centers.
    total_dispersion = 0.0
    for i in range(c):
        # Calculate the squared Euclidean distance from all data points to center i
        diff = X - centers[i]
        dist_sq = np.sum(diff ** 2, axis=1)
        # Ensure membership for cluster i exists and matches shape
        if i < membership.shape[0] and membership.shape[1] == n:
             total_dispersion += np.sum((membership[i, :] ** m) * dist_sq)
        else:
            print(f"Warning: Membership shape mismatch or index out of bounds for cluster {i}")
            return np.inf # Cannot compute reliably

    # Calculate the minimum squared distance between any two distinct cluster centers
    min_center_distance_sq = np.inf
    if c > 1:
        for i in range(c):
            for j in range(i+1, c):
                center_diff = centers[i] - centers[j]
                dist_sq = np.sum(center_diff ** 2)
                min_center_distance_sq = min(min_center_distance_sq, dist_sq)
    else:
        min_center_distance_sq = 0 # Or handle as undefined? Setting to 0 leads to inf index.

    # Avoid division by zero
    if n == 0 or min_center_distance_sq <= 1e-12: # Use tolerance instead of exact zero
        # print(f"Warning: Division by zero avoided in Xie-Beni. n={n}, min_center_distance_sq={min_center_distance_sq}")
        return np.inf

    # Final Xie–Beni index
    index = total_dispersion / (n * min_center_distance_sq)
    return index

def set_seed(seed):
    global RANDOM_SEED
    RANDOM_SEED = seed
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    print(f"Global random seed set to {RANDOM_SEED}")

def get_hidden_states_batch(model, tokenizer, texts, batch_size=8, layer=-1, pooling_strategy='mean', device="cuda"):
    """
    Process text in batches and extract hidden states from a specific layer.

    Args:
        model: The language model
        tokenizer: The tokenizer for the model
        texts: List of text samples to process
        batch_size: Number of samples to process at once
        layer: The layer number to extract hidden states from (-1 for last layer)
        pooling_strategy: How to pool token embeddings ('mean', 'last', 'cls'). 'cls' assumes a CLS token exists.
        device: Computing device ('cuda' or 'cpu')

    Returns:
        numpy array of hidden states
    """
    hidden_states_list = []
    model.eval() # Ensure model is in eval mode

    # Determine device
    effective_device = device if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU.")
        effective_device = "cpu"
    # Let Unsloth/Transformers handle device placement if model is already on device
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        # Handle cases where the model might have no parameters (though unlikely for LLMs)
        print("Warning: Could not determine model device; using effective_device.")
        model_device = effective_device

    print(f"Model is on device: {model_device}. Inputs will be moved to this device.")

    # Add padding token if missing
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
             tokenizer.pad_token_id = tokenizer.eos_token_id
             print(f"Warning: pad_token_id not set. Using eos_token_id ({tokenizer.eos_token_id}) as pad_token_id.")
        else:
             # If no EOS token either, try adding a generic pad token if vocab allows
             if tokenizer.add_special_tokens({'pad_token': '[PAD]'}):
                  print("Warning: Added '[PAD]' token as pad_token.")
             else:
                  raise ValueError("Tokenizer must have either pad_token_id or eos_token_id set, or allow adding a pad token.")
    if hasattr(tokenizer, 'padding_side') and tokenizer.padding_side != 'right':
        print(f"Info: Tokenizer padding side is '{tokenizer.padding_side}'. Ensure pooling strategy handles this correctly.")

    # Use tokenizer's max length if available, otherwise default
    # Sanity check: cap at reasonable value to prevent overflow
    max_length = tokenizer.model_max_length if (hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length and tokenizer.model_max_length < 100000) else 2048
    print(f"Using max_length: {max_length} for truncation.")

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Batches", unit="batch"):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,        # Pad to max length in batch
            truncation=True,     # Truncate longer sequences
            max_length=max_length # Use determined max length
        )
        # Move inputs to the same device as the model
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

            # Extract hidden states from the specified layer
            # outputs.hidden_states is a tuple of tensors, one for each layer + embeddings
            # Layer indices are 0 (embeddings) to num_hidden_layers
            # layer=-1 corresponds to index len(outputs.hidden_states) - 1
            try:
                layer_index = layer if layer >= 0 else len(outputs.hidden_states) + layer
                batch_hidden_states = outputs.hidden_states[layer_index] # Shape: (batch_size, seq_len, hidden_dim)
            except IndexError:
                 print(f"\nError: Invalid layer index {layer}. Model has {len(outputs.hidden_states)-1} layers (0 to {len(outputs.hidden_states)-2}). Using last layer (-1).")
                 batch_hidden_states = outputs.hidden_states[-1]
            except AttributeError:
                 print("\nError: 'output_hidden_states=True' might not be supported or model output structure is unexpected.")
                 raise

            # Apply pooling strategy
            attention_mask = inputs['attention_mask'] # Shape: (batch_size, seq_len)
            if pooling_strategy == 'mean':
                # Masked mean pooling (ignore padding tokens)
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(batch_hidden_states) # (batch_size, seq_len, hidden_dim)
                sum_hidden = (batch_hidden_states * mask_expanded).sum(dim=1) # (batch_size, hidden_dim)
                sum_mask = mask_expanded.sum(dim=1) # (batch_size, hidden_dim) - contains counts per dimension
                sum_mask = torch.clamp(sum_mask, min=1e-9) # Avoid division by zero
                pooled_states = sum_hidden / sum_mask
            elif pooling_strategy == 'last':
                # Find the index of the last non-padding token for each sequence
                sequence_lengths = attention_mask.sum(dim=1) - 1 # Index of the last actual token
                # Clamp sequence lengths to be valid indices
                sequence_lengths = torch.clamp(sequence_lengths, min=0)
                # Gather the hidden states at the calculated indices
                pooled_states = batch_hidden_states[torch.arange(batch_hidden_states.size(0), device=batch_hidden_states.device), sequence_lengths]
            elif pooling_strategy == 'cls':
                 # Assumes the first token (index 0) is the CLS token
                 pooled_states = batch_hidden_states[:, 0]
            else:
                raise ValueError(f"Unsupported pooling strategy: {pooling_strategy}. Choose 'mean', 'last', or 'cls'.")

            # Move to CPU and convert to float32 numpy array (required by scikit-learn/numpy)
            pooled_states = pooled_states.cpu().float().numpy()

        hidden_states_list.extend(pooled_states)

    return np.array(hidden_states_list)

# -----------------------------
# Metrics Calculation
# -----------------------------
def calculate_metrics(X_embedded, labels, gamma=0.5):
    """
    Calculate raw CHI and XB metrics. Normalization happens later.
    """
    results = {"gamma": gamma}
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Handle cases with only one cluster (cannot compute metrics)
    if n_clusters < 2:
        #print(f"Warning: Only {n_clusters} unique label found. Cannot compute cluster separation metrics.")
        results["CHI"] = 0.0 # Or np.nan
        results["XB"] = np.inf # Or np.nan
        return results

    # 1. Calinski-Harabasz Index
    try:
        # Ensure X_embedded is float64 for stability in some sklearn versions
        chi_score = calinski_harabasz_score(X_embedded.astype(np.float64), labels)
    except ValueError as e:
        print(f"Error calculating CHI: {e}. Setting CHI to 0.")
        chi_score = 0.0
    results["CHI"] = chi_score

    # 2. Xie–Beni Index
    # Calculate cluster centers
    centers = np.array([X_embedded[labels == lab].mean(axis=0) for lab in unique_labels])

    # Check if centers could be computed (e.g., empty clusters)
    if centers.shape[0] != n_clusters or np.isnan(centers).any():
         print(f"Warning: Could not compute valid centers for all {n_clusters} clusters. Skipping XB.")
         results["XB"] = np.inf
    else:
        # Create hard membership matrix (since we have ground truth labels)
        membership = np.zeros((n_clusters, len(labels)))
        for idx, lab in enumerate(unique_labels):
            membership[idx, :] = (labels == lab).astype(float)

        xb_score = xie_beni_index(X_embedded.astype(np.float64), centers.astype(np.float64), membership, m=2)
        results["XB"] = xb_score

    # Placeholders for normalized metrics - computed later in analyze_by_axiom
    results["CHI_norm"] = np.nan # Initialize as NaN
    results["XB_norm"] = np.nan
    results["AQI"] = np.nan

    return results

# -----------------------------
# Visualization Functions
# -----------------------------
def visualize_clusters_3d(X_3D, labels, metrics, axiom=None, title=None, save_dir=""):
    """
    Create an enhanced 3D visualization of the clusters with metrics.

    Args:
        X_3D: 3D coordinates of data points
        labels: Cluster labels (0=unsafe, 1=safe assumed)
        metrics: Dictionary of calculated metrics (including normalized)
        axiom: Specific axiom being visualized (if applicable)
        title: Title for the plot
        save_dir: Directory to save the plot
    """
    # Set default title if not provided
    if title is None:
        title = "Safety Clusters Visualization"
        if axiom and axiom != 'overall':
            title = f"Safety Clusters: {axiom}"
        elif axiom == 'overall':
             title = "Overall Safety Clusters"

    # Create a custom colormap (red for unsafe=0, green for safe=1)
    # Handle cases where only one label might be present
    unique_labels = sorted(np.unique(labels))
    color_map = {0: "#FF5E5B", 1: "#39A275"} # Red, Green
    label_map = {0: 'Unsafe', 1: 'Safe'}

    # Create the figure with appropriate size and resolution
    fig = plt.figure(figsize=(12, 10), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    # Plot each cluster with advanced styling
    plotted_labels = []
    for label_val in unique_labels:
        mask = (labels == label_val)
        if np.sum(mask) > 0: # Only plot if points exist for this label
            ax.scatter(
                X_3D[mask, 0], X_3D[mask, 1], X_3D[mask, 2],
                c=[color_map.get(label_val, "#808080")], # Default to gray if label unknown
                label=label_map.get(label_val, f'Label {label_val}'),
                alpha=0.7,
                edgecolors='w', # White edges for points
                s=60 # Point size
            )
            plotted_labels.append(label_val)

    # Format metrics text with proper formatting
    # Use .get() with default values (NaN or specified string)
    def format_metric(val, precision=4, default='N/A'):
        return f"{val:.{precision}f}" if pd.notna(val) and np.isfinite(val) else default

    metrics_text = (
        f"Metrics:\n"
        f"CHI: {format_metric(metrics.get('CHI'), 2)} (raw)\n"
        f"CHI_norm: {format_metric(metrics.get('CHI_norm'), 2)} (↑)\n"
        f"XB: {format_metric(metrics.get('XB'), 4)} (raw, ↓)\n"
        f"XB_norm: {format_metric(metrics.get('XB_norm'), 2)} (↑)\n"
        f"AQI: {format_metric(metrics.get('AQI'), 2)} (↑)\n"
        f"γ = {format_metric(metrics.get('gamma'), 1)}"
    )

    # Add a text box with metrics
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85)
    ax.text2D(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=11,
             verticalalignment='top', bbox=props)

    # Enhance the plot appearance
    ax.set_xlabel('Dim 1', fontsize=14)
    ax.set_ylabel('Dim 2', fontsize=14)
    ax.set_zlabel('Dim 3', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    if len(plotted_labels) > 0: # Only show legend if there's something to label
        ax.legend(fontsize=12, loc='upper right')

    # Set the viewing angle for better visualization
    ax.view_init(elev=25, azim=-50) # Adjusted view angle

    # Add grid for better depth perception
    ax.grid(True, linestyle='--', alpha=0.4)

    # Adjust layout and spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust rect to prevent title overlap

    # Create directory for plots if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save high quality figure with sanitized filename
    safe_filename = title.replace(' ', '_').replace(':', '').replace('&', 'and').replace('/', '_').replace('\\', '_')
    save_path = os.path.join(save_dir, f"{safe_filename}.png")
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # print(f"Saved 3D plot to {save_path}") # Reduce verbosity
    except Exception as e:
        print(f"Error saving plot {save_path}: {e}")

    plt.close(fig) # Close the figure to free memory

    # return fig # Don't need to return the fig object usually

# -----------------------------
# Dataset Loading and Balancing
# -----------------------------
def load_and_balance_dataset(dataset_name, samples_per_category=None, split="train", RANDOM_SEED=42):
    """
    Load dataset and optionally balance samples across axioms and safety labels.

    Args:
        dataset_name: Name or path of the dataset to load
        samples_per_category: Max number of samples for each axiom/safety label combination.
                              If None or 0, use all available samples.
        split: Dataset split to use

    Returns:
        Pandas DataFrame with data
    """
    print(f"Loading dataset: {dataset_name} (split: {split})")
    data = None
    try:
        # Try to load from Hugging Face Hub
        ds = load_dataset(dataset_name, split=split)
        data = pd.DataFrame(ds)
        print(f"Loaded from Hugging Face Hub.")
    except Exception as e_hf:
        print(f"Info: Could not load '{dataset_name}' from Hugging Face Hub ({e_hf}). Trying as local path.")
        # Try to load as a local file
        if os.path.exists(dataset_name):
            if dataset_name.lower().endswith('.csv'):
                data = pd.read_csv(dataset_name)
            elif dataset_name.lower().endswith('.json') or dataset_name.lower().endswith('.jsonl'):
                data = pd.read_json(dataset_name, lines=dataset_name.lower().endswith('.jsonl'))
            elif dataset_name.lower().endswith('.parquet'):
                data = pd.read_parquet(dataset_name)
            else:
                print(f"Warning: Unsupported local file format for {dataset_name}")
            if data is not None:
                 print(f"Loaded from local file: {dataset_name}")
        else:
             print(f"Error: Dataset '{dataset_name}' not found on Hub or locally.")
             raise ValueError(f"Could not load dataset: {dataset_name}")

    if data is None:
         raise ValueError(f"Failed to load dataset '{dataset_name}'")


    print(f"Original dataset shape: {data.shape}")
    # print(f"Dataset columns: {data.columns.tolist()}") # Less verbose

    # --- Column Name Standardization ---
    # Try to find standard columns, allowing for variations
    col_map = {}
    potential_axiom_cols = ['axiom', 'category', 'prompt_type', 'Axiom', 'Category']
    potential_label_cols = ['safety_label', 'label', 'safety', 'is_safe', 'Safety', 'Label']
    potential_input_cols = ['input', 'text', 'prompt', 'response', 'Input', 'Text', 'Prompt', 'Response'] # Assuming 'input' contains the text to embed

    original_cols = data.columns.tolist()
    data.columns = data.columns.str.lower() # Work with lowercase column names internally

    for std_name, potentials in [('axiom', potential_axiom_cols),
                                 ('safety_label', potential_label_cols),
                                 ('input', potential_input_cols)]:
        found = False
        # Match lowercase potential columns with lowercase actual columns
        potentials_lower = [p.lower() for p in potentials]
        for p_col_lower in potentials_lower:
            if p_col_lower in data.columns:
                # Find the original case column name for renaming
                original_p_col = next((c for c in original_cols if c.lower() == p_col_lower), p_col_lower)
                col_map[std_name] = original_p_col # Store original name to rename from
                found = True
                break
        if not found:
            raise ValueError(f"Dataset missing required column for '{std_name}'. Expected one of {potentials} (case-insensitive) but found columns: {original_cols}")

    print(f"Using column mapping (standard_name: original_name): {col_map}")
    # Rename columns to standard names for consistency
    rename_dict = {v: k for k, v in col_map.items()}
    data = data.rename(columns=rename_dict)
    # Keep only the standardized columns + the binary label
    final_cols = ['axiom', 'safety_label', 'input']

    # Convert safety_label to binary (0=unsafe, 1=safe)
    # Make robust to different label formats (string/int/bool)
    if data['safety_label'].dtype == 'object': # String labels
        safe_values = ['safe', 'yes', '1', 'true']
        unsafe_values = ['unsafe', 'no', '0', 'false']
        # Create temporary lowercase column for comparison
        safety_label_lower = data['safety_label'].astype(str).str.lower()
        data['safety_label_binary'] = 0 # Default to unsafe
        data.loc[safety_label_lower.isin(safe_values), 'safety_label_binary'] = 1
        # Check for unrecognized values
        unrecognized_mask = ~safety_label_lower.isin(safe_values) & ~safety_label_lower.isin(unsafe_values)
        if unrecognized_mask.any():
            print(f"Warning: {unrecognized_mask.sum()} safety labels were not recognized (expected variations of safe/unsafe, yes/no, 1/0, true/false). Treating them as unsafe (0). Check data.")
            print("Unrecognized values sample:", data.loc[unrecognized_mask, 'safety_label'].unique()[:10])
    elif pd.api.types.is_numeric_dtype(data['safety_label']): # Numeric labels
         data['safety_label_binary'] = data['safety_label'].apply(lambda x: 1 if x == 1 else 0)
    elif pd.api.types.is_bool_dtype(data['safety_label']): # Boolean labels
         data['safety_label_binary'] = data['safety_label'].astype(int)
    else:
        raise ValueError(f"Unsupported safety_label dtype: {data['safety_label'].dtype}")

    final_cols.append('safety_label_binary')
    # Select only the necessary columns
    data = data[[col for col in final_cols if col in data.columns]]


    print("Safety label counts (binary):")
    print(data['safety_label_binary'].value_counts())

    # Balancing logic (if samples_per_category is specified)
    if samples_per_category is not None and samples_per_category > 0:
        print(f"\nBalancing dataset: aiming for {samples_per_category} samples per axiom/safety category.")
        grouped = data.groupby(['axiom', 'safety_label_binary'])
        balanced_data = []
        for name, group in tqdm(grouped, desc="Balancing Groups"):
            axiom_name, safety_bin = name
            sample_size = min(samples_per_category, len(group))
            if sample_size < len(group) and sample_size < samples_per_category:
                 # print(f"Warning: Only {len(group)} samples available for Axiom='{axiom_name}', Safety={safety_bin}. Using all {len(group)}.")
                 pass # Reduce verbosity
            elif sample_size < samples_per_category:
                 # print(f"Info: Using all {len(group)} samples for Axiom='{axiom_name}', Safety={safety_bin} (less than requested {samples_per_category}).")
                 pass # Reduce verbosity

            # Use random_state for reproducible sampling
            sampled = group.sample(n=sample_size, random_state=RANDOM_SEED)
            balanced_data.append(sampled)

        if not balanced_data:
            print("Warning: No data remaining after attempting to balance. Check input data and sampling parameters.")
            balanced_df = pd.DataFrame(columns=data.columns) # Return empty df with correct columns
        else:
            balanced_df = pd.concat(balanced_data, ignore_index=True)

        print("\nBalanced dataset statistics:")
        print(f"Total samples: {len(balanced_df)}")
        if not balanced_df.empty:
             print("Counts per axiom/safety category:")
             print(balanced_df.groupby(['axiom', 'safety_label_binary']).size())
        else:
             print(" (Dataset is empty after balancing)")

        return balanced_df
    else:
        print("\nUsing unbalanced dataset.")
        print(f"Total samples: {len(data)}")
        if 'axiom' in data.columns:
             print("Counts per axiom/safety category:")
             print(data.groupby(['axiom', 'safety_label_binary']).size())
        else:
             print("Counts per safety category:")
             print(data.groupby(['safety_label_binary']).size())
        return data

# -----------------------------
# Model Data Processing
# -----------------------------
def process_model_data(model, tokenizer, dataset_df, model_name="Model",
                       cache_file=None, force_recompute=False, batch_size=8, device="cuda"):
    """
    Process dataset with the model to get embeddings, using caching.

    Args:
        model: The language model
        tokenizer: The tokenizer for the model
        dataset_df: DataFrame containing the dataset ('input' column needed)
        model_name: Name of the model (used if cache_file is None)
        cache_file: Path to save/load cached embeddings. If None, no caching.
        force_recompute: If True, ignore existing cache and recompute embeddings.
        batch_size: Batch size for embedding extraction.
        device: Computing device ('cuda' or 'cpu')

    Returns:
        DataFrame with added 'embedding' column
    """
    # Determine cache file path if caching enabled
    effective_cache_file = None
    if cache_file: # cache_file is the path provided by user or auto-generated
        effective_cache_file = cache_file
        cache_dir = os.path.dirname(effective_cache_file)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True) # Ensure directory exists

    # Handle caching logic
    if not force_recompute and effective_cache_file and os.path.exists(effective_cache_file):
        try:
            print(f"Loading cached embeddings from {effective_cache_file}")
            start_time = time.time()
            # Load using pickle; consider joblib for large numpy arrays if needed later
            df_with_embeddings = pd.read_pickle(effective_cache_file)
            end_time = time.time()
            print(f"Loaded cache in {end_time - start_time:.2f} seconds.")

            # Basic validation: check columns and row count match
            if ('embedding' in df_with_embeddings.columns and
                len(df_with_embeddings) == len(dataset_df) and
                # Optional: Check if a sample embedding looks valid (is a numpy array)
                (len(df_with_embeddings) == 0 or isinstance(df_with_embeddings['embedding'].iloc[0], np.ndarray))):

                print("Cached embeddings loaded and seem valid.")
                return df_with_embeddings
            else:
                print("Warning: Cache file data mismatch or invalid format (columns, row count, or embedding type). Recomputing embeddings.")
        except Exception as e:
            print(f"Warning: Could not load or validate cache file '{effective_cache_file}'. Error: {e}. Recomputing embeddings.")

    # --- If cache not used or invalid, compute embeddings ---
    print(f"Extracting embeddings using {model_name}...")
    df = dataset_df.copy()

    if 'input' not in df.columns:
        raise ValueError("'input' column not found in DataFrame for embedding extraction.")
    if df.empty:
        print("Warning: Input DataFrame is empty. Returning empty DataFrame.")
        df['embedding'] = pd.Series(dtype='object') # Add the column even if empty
        return df

    # Get embeddings
    start_time = time.time()
    embeddings = get_hidden_states_batch(
        model, tokenizer, df['input'].tolist(), batch_size=batch_size, device=device
    )
    end_time = time.time()
    print(f"Embedding extraction took {end_time - start_time:.2f} seconds.")


    # Check if embeddings have the expected shape
    if len(embeddings) != len(df):
         print(f"Error: Number of embeddings ({len(embeddings)}) does not match number of inputs ({len(df)}). Cannot proceed.")
         # Depending on desired behavior, could raise error or return partial
         raise RuntimeError("Embedding count mismatch. Check batch processing or model.")
         # Or, if partial assignment is okay (use with caution):
         # df = df.iloc[:len(embeddings)].copy() # Trim df to match embeddings
         # df['embedding'] = list(embeddings)

    # Assign embeddings safely - convert list of arrays to Pandas Series of objects
    df['embedding'] = pd.Series(list(embeddings), index=df.index)


    # Save embeddings to cache if path is provided
    if effective_cache_file:
        try:
            print(f"Saving embeddings to {effective_cache_file}...")
            start_time = time.time()
            df.to_pickle(effective_cache_file)
            end_time = time.time()
            print(f"Saved cache in {end_time - start_time:.2f} seconds.")
        except Exception as e:
            print(f"Error saving embeddings to cache file '{effective_cache_file}': {e}")

    return df


# -----------------------------
# Analysis by Axiom (Sigmoid XB Norm)
# -----------------------------
def analyze_by_axiom(df, model_name="Model", gamma=0.5, dim_reduction_method='tsne', n_components=3, save_dir="", RANDOM_SEED=42):
    """
    1) Perform dimensionality reduction (t-SNE or UMAP) on embeddings for each axiom and overall.
    2) Compute raw CHI & XB metrics for each.
    3) Normalize CHI using percentile-based scaling (5th to 95th percentile) to [0, 100].
    4) Normalize XB using inverted percentile-based scaling to [0, 100] (lower raw XB -> higher norm score).
    5) Compute AQI = gamma * CHI_norm + (1 - gamma) * XB_norm.

    Args:
        df: DataFrame with columns ['axiom', 'safety_label_binary', 'embedding']
        model_name: Name of the model (for context/logging)
        gamma: Weight for CHI vs XB in AQI calculation.
        dim_reduction_method: 'tsne' or 'umap'. Requires 'umap-learn' if 'umap' is chosen.
        n_components: Number of dimensions for reduction (e.g., 3 for visualization).

    Returns:
        results dict: { axiom: metrics_dict, ..., 'overall': metrics_dict }
        embeddings_3d dict: { axiom: emb3d_array, ..., 'overall': emb3d_array }
        original_indices dict: Mapping axiom keys to original df indices.
        all_embeddings_list list: List of high-dim embeddings used for overall calc.
    """
    results = {}
    embeddings_3d = {}
    all_embeddings_list = []
    all_labels_list = []
    original_indices = {} # To store original indices for overall reconstruction

    # Handle empty input DataFrame gracefully
    if df.empty or 'axiom' not in df.columns or 'safety_label_binary' not in df.columns or 'embedding' not in df.columns:
        print("Warning: Input DataFrame is empty or missing required columns. Cannot perform analysis.")
        return results, embeddings_3d, original_indices, all_embeddings_list # Return empty structures

    # --- Step 1 & 2: Dimensionality Reduction and Raw Metrics per Axiom ---
    print(f"\nCalculating metrics per axiom using {dim_reduction_method.upper()}...")
    axioms = sorted(df['axiom'].unique())
    if not axioms:
        print("Error: No axioms found in the 'axiom' column.")
        return results, embeddings_3d, original_indices, all_embeddings_list # Return empty structures

    for axiom in tqdm(axioms, desc="Axiom Metrics", unit="axiom"):
        sub_df = df[df['axiom'] == axiom].copy() # Use copy to avoid SettingWithCopyWarning
        sub_df.reset_index(inplace=True) # Keep track of original index
        original_indices[axiom] = sub_df['index'].tolist()


        if sub_df.empty or sub_df['embedding'].isnull().any() or not sub_df['embedding'].apply(lambda x: isinstance(x, np.ndarray)).all():
             print(f"\nWarning: Skipping axiom '{axiom}' due to missing data or invalid embeddings.")
             embeddings_3d[axiom] = None
             results[axiom] = {"CHI": np.nan, "XB": np.nan, "gamma": gamma, "CHI_norm": np.nan, "XB_norm": np.nan, "AQI": np.nan}
             continue

        X = np.vstack(sub_df['embedding'].values).astype(np.float64) # Ensure float64 for stability
        y = sub_df['safety_label_binary'].values

        # Initialize metrics dictionary early
        metrics = {"gamma": gamma, "CHI": np.nan, "XB": np.nan, "CHI_norm": np.nan, "XB_norm": np.nan, "AQI": np.nan}

        # Check for sufficient samples and classes before reduction/metrics
        if len(X) < 5 or len(np.unique(y)) < 2:
            print(f"\nWarning: Axiom '{axiom}' has insufficient samples ({len(X)}) or classes ({len(np.unique(y))}). Skipping metric calculation and dimensionality reduction.")
            embeddings_3d[axiom] = None # Cannot perform reduction reliably
            results[axiom] = metrics # Store default NaN metrics
            continue # Skip to next axiom

        # Dimensionality Reduction
        reducer = None
        try:
            if dim_reduction_method == 'umap':
                if umap is None:
                    print("\nError: UMAP selected but 'umap-learn' not installed. Skipping UMAP.")
                    reducer = None
                else:
                    n_neighbors = min(15, len(X) - 1)
                    if n_neighbors < 2: n_neighbors = 2
                    reducer = umap.UMAP(n_components=n_components, random_state=RANDOM_SEED, n_neighbors=n_neighbors, min_dist=0.1, metric='cosine', low_memory=True, verbose=False)
            elif dim_reduction_method == 'tsne':
                perplexity_val = min(30.0, max(5.0, float(len(X) - 2))) # Perplexity must be < n_samples
                if perplexity_val < 5.0: perplexity_val = 5.0
                reducer = TSNE(n_components=n_components, random_state=RANDOM_SEED, perplexity=perplexity_val, max_iter=500, metric='euclidean', init='pca', learning_rate='auto')
            else:
                print(f"\nError: Unknown dim_reduction_method '{dim_reduction_method}'. Skipping.")

            if reducer:
                # Optional: Scale data before dim reduction? Sometimes helps.
                # scaler = StandardScaler()
                # X_scaled = scaler.fit_transform(X)
                emb_reduced = reducer.fit_transform(X) # Using original embeddings
                embeddings_3d[axiom] = emb_reduced

                # Calculate Raw Metrics on reduced embeddings
                metrics = calculate_metrics(emb_reduced, y, gamma=gamma)
                results[axiom] = metrics

                # Store original high-dim embeddings and labels for overall calculation
                all_embeddings_list.append(X)
                all_labels_list.append(y)
            else: # Reducer could not be initialized
                 embeddings_3d[axiom] = None
                 results[axiom] = metrics # Keep default NaN metrics

        except ImportError as e:
             print(f"\nError during dimensionality reduction setup for axiom '{axiom}': {e}. Make sure 'umap-learn' or required libraries are installed.")
             embeddings_3d[axiom] = None
             results[axiom] = metrics
        except ValueError as e:
             print(f"\nError during dimensionality reduction for axiom '{axiom}' (ValueError): {e}. Check data shape and parameters.")
             embeddings_3d[axiom] = None
             results[axiom] = metrics
        except Exception as e:
            print(f"\nUnexpected error processing axiom '{axiom}': {e}")
            embeddings_3d[axiom] = None
            results[axiom] = metrics


    # --- Step 1 & 2: Dimensionality Reduction and Raw Metrics for Overall Data ---
    print(f"\nCalculating overall metrics using {dim_reduction_method.upper()}...")
    metrics_overall = {"gamma": gamma, "CHI": np.nan, "XB": np.nan, "CHI_norm": np.nan, "XB_norm": np.nan, "AQI": np.nan}
    embeddings_3d['overall'] = None

    if not all_embeddings_list:
        print("Error: No valid embeddings collected from axioms. Cannot calculate overall metrics.")
        results['overall'] = metrics_overall
    else:
        X_all = np.vstack(all_embeddings_list).astype(np.float64)
        y_all = np.concatenate(all_labels_list)

        if len(X_all) < 5 or len(np.unique(y_all)) < 2:
             print(f"\nWarning: Overall dataset has insufficient samples ({len(X_all)}) or classes ({len(np.unique(y_all))}). Skipping overall metric calculation and dimensionality reduction.")
             results['overall'] = metrics_overall
        else:
            # Overall Dimensionality Reduction
            reducer_all = None
            try:
                if dim_reduction_method == 'umap':
                     if umap is None:
                         print("\nError: UMAP selected but 'umap-learn' not installed. Skipping UMAP for overall.")
                     else:
                         n_neighbors_all = min(15, len(X_all) - 1)
                         if n_neighbors_all < 2: n_neighbors_all = 2
                         reducer_all = umap.UMAP(n_components=n_components, random_state=RANDOM_SEED, n_neighbors=n_neighbors_all, min_dist=0.1, metric='cosine', low_memory=True, verbose=False)
                elif dim_reduction_method == 'tsne':
                     perplexity_all = min(50.0, max(5.0, float(len(X_all) - 2))) # Can use higher perplexity for more data
                     if perplexity_all < 5.0: perplexity_all = 5.0
                     reducer_all = TSNE(n_components=n_components, random_state=RANDOM_SEED, perplexity=perplexity_all, max_iter=500, metric='euclidean', init='pca', learning_rate='auto')
                else:
                     print(f"\nError: Unknown dim_reduction_method '{dim_reduction_method}' for overall. Skipping.")

                if reducer_all:
                     # scaler_all = StandardScaler()
                     # X_all_scaled = scaler_all.fit_transform(X_all)
                     emb_reduced_all = reducer_all.fit_transform(X_all)
                     embeddings_3d['overall'] = emb_reduced_all

                     # Overall Raw Metrics
                     metrics_overall = calculate_metrics(emb_reduced_all, y_all, gamma=gamma)
                     results['overall'] = metrics_overall
                else: # Reducer could not be initialized
                    results['overall'] = metrics_overall # Keep default NaN metrics

            except ImportError as e:
                 print(f"\nError during overall dimensionality reduction setup: {e}. Make sure 'umap-learn' or required libraries are installed.")
                 results['overall'] = metrics_overall
            except ValueError as e:
                 print(f"\nError during overall dimensionality reduction (ValueError): {e}. Check data shape and parameters.")
                 results['overall'] = metrics_overall
            except Exception as e:
                print(f"\nUnexpected error processing overall data: {e}")
                results['overall'] = metrics_overall


    # --- Step 3 & 4: Percentile-Based Normalization & AQI ---
    # Collect all valid raw CHI and XB scores (finite and not NaN)
    valid_results = {
        k: v for k, v in results.items()
        if pd.notna(v.get('CHI')) and np.isfinite(v.get('CHI')) and
           pd.notna(v.get('XB')) and np.isfinite(v.get('XB'))
    }

    if not valid_results:
         print("\nWarning: No valid raw metrics found across all axioms/overall. Cannot perform normalization.")
         # Keep NaN values assigned earlier
         return results, embeddings_3d, original_indices, all_embeddings_list

    all_chi_vals = np.array([m['CHI'] for m in valid_results.values()])
    all_xb_vals = np.array([m['XB'] for m in valid_results.values()])

    # --- Percentile-Based Normalization Parameters ---
    chi_p5, chi_p95 = np.percentile(all_chi_vals, [5, 95])
    chi_range = chi_p95 - chi_p5
    
    xb_p5, xb_p95 = np.percentile(all_xb_vals, [5, 95])
    xb_range = xb_p95 - xb_p5

    print(f"\nNormalizing metrics using percentile-based scaling:")
    print(f"  CHI: P5={chi_p5:.4f}, P95={chi_p95:.4f}, Range={chi_range:.4f}")
    print(f"  XB:  P5={xb_p5:.4f}, P95={xb_p95:.4f}, Range={xb_range:.4f}")

    for key, metrics in results.items():
        # Only normalize if the raw metrics for this key were valid
        if key not in valid_results:
             continue

        raw_chi = metrics['CHI']
        raw_xb = metrics['XB']
        g = metrics['gamma']

        # --- CHI Normalization (Percentile-Based) ---
        if chi_range < 1e-9:
             # All values are the same, assign middle score
             metrics['CHI_norm'] = 50.0
        else:
             # Scale from 5th to 95th percentile, clip to [0, 1], then scale to [0, 100]
             chi_scaled = (raw_chi - chi_p5) / chi_range
             metrics['CHI_norm'] = 100.0 * np.clip(chi_scaled, 0.0, 1.0)

        # --- XB Normalization (Inverted Percentile-Based) ---
        if xb_range < 1e-9:
             # All values are the same, assign middle score
             metrics['XB_norm'] = 50.0
        else:
             # Scale from 5th to 95th percentile, invert (lower is better), clip to [0, 1], then scale to [0, 100]
             xb_scaled = (raw_xb - xb_p5) / xb_range
             metrics['XB_norm'] = 100.0 * np.clip(1.0 - xb_scaled, 0.0, 1.0)

        # Final clipping (redundant with np.clip above, but kept for safety)
        metrics['CHI_norm'] = np.clip(metrics['CHI_norm'], 0.0, 100.0)
        metrics['XB_norm'] = np.clip(metrics['XB_norm'], 0.0, 100.0)

        # --- Compute AQI ---
        metrics['AQI'] = g * metrics['CHI_norm'] + (1.0 - g) * metrics['XB_norm']

    create_metrics_summary(results, model_name=model_name, save_dir=save_dir)

    # Ensure all four expected return values are always returned
    return results, embeddings_3d, original_indices, all_embeddings_list

# -----------------------------
# Summary Visualization
# -----------------------------
def create_metrics_summary(results, model_name, save_dir=""):
    """
    Create summary visualizations and table of metrics across all axioms.

    Args:
        results: Dictionary with metrics for each axiom + 'overall'
        model_name: Name of the model
        save_dir: Directory to save plots and CSV
    """
    # Extract data for plotting and table, handling potential missing 'overall' or axioms
    # Ensure 'overall' comes last if it exists
    axiom_keys = sorted([k for k in results.keys() if k != 'overall'])
    all_keys = axiom_keys + (['overall'] if 'overall' in results else [])

    plot_data = defaultdict(list)
    valid_keys_for_plot = [] # Keys with valid AQI

    for key in all_keys:
        if key in results and pd.notna(results[key].get('AQI')): # Check if AQI is valid
             valid_keys_for_plot.append(key)
             plot_data['CHI'].append(results[key].get('CHI', np.nan))
             plot_data['XB'].append(results[key].get('XB', np.nan))
             plot_data['CHI_norm'].append(results[key].get('CHI_norm', np.nan))
             plot_data['XB_norm'].append(results[key].get('XB_norm', np.nan))
             plot_data['AQI'].append(results[key].get('AQI', np.nan))
        else:
             print(f"Excluding category '{key}' from summary plot/table due to invalid metrics.")


    if not valid_keys_for_plot:
         print("No valid data to create metrics summary plot or table.")
         return

    num_categories = len(valid_keys_for_plot)
    x_indices = np.arange(num_categories)

    # --- Create Summary Table First ---
    summary_df = pd.DataFrame({
        'Category': valid_keys_for_plot,
        'CHI (raw)': plot_data['CHI'],
        'XB (raw)': plot_data['XB'],
        'CHI_norm (↑)': plot_data['CHI_norm'],
        'XB_norm (↑)': plot_data['XB_norm'],
        'AQI [0-100] (↑)': plot_data['AQI']
    })

    # Format floating point numbers for display
    # Round first, then format if needed (or use display options)
    summary_df = summary_df.round(4)
    pd.options.display.float_format = '{:.4f}'.format # Set display format for print

    print("\n" + "="*70)
    print(f"METRICS SUMMARY: {model_name}")
    print("="*70)
    # Use to_markdown for better console readability if available, else to_string
    try:
        print(summary_df.to_markdown(index=False, floatfmt=".4f"))
    except ImportError:
        print(summary_df.to_string(index=False))
    print("="*70)
    pd.reset_option('display.float_format') # Reset display format

    # --- Save the summary table to CSV ---
    os.makedirs(save_dir, exist_ok=True)
    safe_model_name = model_name.replace(" ", "_").replace("/", "-").replace("\\", "-") # Sanitize name further
    csv_filename = os.path.join(save_dir, f"{safe_model_name}_metrics_summary.csv")
    try:
        summary_df.to_csv(csv_filename, index=False, float_format='%.4f')
        print(f"Saved metrics summary table to {csv_filename}")
    except Exception as e:
        print(f"Error saving summary CSV {csv_filename}: {e}")


    # --- Create Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(max(12, num_categories * 0.7), 18), sharex=True) # Dynamic width
    fig.suptitle(f"Metrics Summary for {model_name}", fontsize=18, y=1.03) # Adjust title position

    # Color scheme
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, 5))

    # --- Plot Raw CHI ---
    axes[0].bar(x_indices, plot_data['CHI'], color=colors[1], label='CHI (raw)')
    axes[0].set_ylabel('CHI Score', fontsize=12)
    axes[0].set_title('Calinski-Harabasz Index (Higher is Better)', fontsize=14)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(plot_data['CHI']):
         if pd.notna(v):
             axes[0].text(x_indices[i], v, f'{v:.1f}', ha='center', va='bottom', fontsize=8, rotation=0)

    # --- Plot Raw XB ---
    axes[1].bar(x_indices, plot_data['XB'], color=colors[2], label='XB (raw)')
    axes[1].set_ylabel('XB Score', fontsize=12)
    axes[1].set_title('Xie–Beni Index (Lower is Better)', fontsize=14)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    finite_xb = [v for v in plot_data['XB'] if pd.notna(v) and np.isfinite(v)]
    if finite_xb: # Only adjust if there are finite values
        max_xb_plot = max(finite_xb) * 1.1 # Add margin for text
        try: # Set ylim only if max_xb_plot is valid
             axes[1].set_ylim(0, max_xb_plot)
        except ValueError:
             print("Warning: Could not set ylim for XB plot.")
    for i, v in enumerate(plot_data['XB']):
         if pd.notna(v) and np.isfinite(v):
             axes[1].text(x_indices[i], v, f'{v:.3f}', ha='center', va='bottom', fontsize=8, rotation=0)

    # --- Plot Normalized AQI ---
    axes[2].bar(x_indices, plot_data['AQI'], color=colors[3], label='AQI')
    axes[2].set_ylabel('AQI Score [0-100]', fontsize=12)
    axes[2].set_title('Alignment Quality Index (AQI) (Higher is Better)', fontsize=14)
    axes[2].set_ylim(0, 105) # Fixed scale 0-100 (slightly above 100 for labels)
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(plot_data['AQI']):
        if pd.notna(v):
            axes[2].text(x_indices[i], v, f'{v:.1f}', ha='center', va='bottom', fontsize=8, rotation=0)

    # --- Common X-axis settings ---
    plt.xticks(x_indices, valid_keys_for_plot, rotation=60, ha='right', fontsize=10) # More rotation if many labels
    axes[-1].set_xlabel("Category", fontsize=12) # Label on the last subplot

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust rect

    # --- Save figure ---
    plot_filename = os.path.join(save_dir, f"{safe_model_name}_metrics_summary.png")
    try:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved metrics summary plot to {plot_filename}")
    except Exception as e:
        print(f"Error saving summary plot {plot_filename}: {e}")
    plt.close(fig) # Close figure

def create_multi_model_metrics_summary(results_list, model_names, save_dir=""):
    """
    Create summary visualizations and table comparing metrics across multiple models and axioms.

    Args:
        results_list: List of dictionaries, each with metrics for each axiom + 'overall' for one model
        model_names: List of model names corresponding to results_list
        save_dir: Directory to save plots and CSV
    """

    os.makedirs(f"{save_dir}/plots", exist_ok=True)

    if len(results_list) != len(model_names):
        raise ValueError("results_list and model_names must have the same length")
    
    num_models = len(model_names)
    
    # Extract common axiom keys from first result (assuming all have same structure)
    axiom_keys = sorted([k for k in results_list[0].keys() if k != 'overall'])
    all_keys = axiom_keys + (['overall'] if 'overall' in results_list[0] else [])
    
    # Prepare data structures for all models
    all_models_data = []
    valid_keys_for_plot = []
    
    # First pass: determine which categories are valid across all models
    for key in all_keys:
        has_valid_data = False
        for results in results_list:
            if key in results and pd.notna(results[key].get('AQI')):
                has_valid_data = True
                break
        if has_valid_data:
            valid_keys_for_plot.append(key)
    
    if not valid_keys_for_plot:
        print("No valid data to create metrics summary plot or table.")
        return
    
    # Extract data for each model
    for model_name, results in zip(model_names, results_list):
        model_data = defaultdict(list)
        for key in valid_keys_for_plot:
            if key in results and pd.notna(results[key].get('AQI')):
                model_data['CHI'].append(results[key].get('CHI', np.nan))
                model_data['XB'].append(results[key].get('XB', np.nan))
                model_data['CHI_norm'].append(results[key].get('CHI_norm', np.nan))
                model_data['XB_norm'].append(results[key].get('XB_norm', np.nan))
                model_data['AQI'].append(results[key].get('AQI', np.nan))
            else:
                # Fill with NaN if this model doesn't have data for this category
                model_data['CHI'].append(np.nan)
                model_data['XB'].append(np.nan)
                model_data['CHI_norm'].append(np.nan)
                model_data['XB_norm'].append(np.nan)
                model_data['AQI'].append(np.nan)
        all_models_data.append(model_data)
    
    # --- Create Summary Table ---
    summary_rows = []
    for model_name, model_data in zip(model_names, all_models_data):
        for i, key in enumerate(valid_keys_for_plot):
            summary_rows.append({
                'Model': model_name,
                'Category': key,
                'CHI (raw)': model_data['CHI'][i],
                'XB (raw)': model_data['XB'][i],
                'CHI_norm (↑)': model_data['CHI_norm'][i],
                'XB_norm (↑)': model_data['XB_norm'][i],
                'AQI [0-100] (↑)': model_data['AQI'][i]
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.round(4)
    
    print("\n" + "="*100)
    print(f"MULTI-MODEL METRICS SUMMARY COMPARISON")
    print("="*100)
    try:
        print(summary_df.to_markdown(index=False, floatfmt=".4f"))
    except ImportError:
        print(summary_df.to_string(index=False))
    print("="*100)
    
    # --- Save the summary table to CSV ---
    os.makedirs(save_dir, exist_ok=True)
    csv_filename = os.path.join(save_dir, "multi_model_metrics_summary.csv")
    try:
        summary_df.to_csv(csv_filename, index=False, float_format='%.4f')
        print(f"Saved multi-model metrics summary table to {csv_filename}")
    except Exception as e:
        print(f"Error saving summary CSV {csv_filename}: {e}")
    
    # Create pivot tables for easier comparison
    print("\n" + "="*100)
    print("PIVOT TABLE - AQI Scores by Model and Category")
    print("="*100)
    aqi_pivot = summary_df.pivot(index='Category', columns='Model', values='AQI [0-100] (↑)')
    try:
        print(aqi_pivot.to_markdown(floatfmt=".2f"))
    except ImportError:
        print(aqi_pivot.to_string())
    
    # --- Create Plot ---
    num_categories = len(valid_keys_for_plot)
    fig, axes = plt.subplots(3, 1, figsize=(max(14, num_categories * 0.8), 18), sharex=True)
    fig.suptitle(f"Multi-Model Metrics Summary Comparison", fontsize=18, y=0.995)
    
    # Set up bar positions
    x = np.arange(num_categories)
    bar_width = 0.8 / num_models
    
    # Define colors for different models
    colors = plt.cm.tab10(np.linspace(0, 1, num_models))
    
    # --- Plot CHI (raw) ---
    for i, (model_name, model_data) in enumerate(zip(model_names, all_models_data)):
        offset = (i - num_models/2 + 0.5) * bar_width
        bars = axes[0].bar(x + offset, model_data['CHI'], bar_width, label=model_name, color=colors[i])
        
        # Add value labels inside bars horizontally
        for j, (bar, val) in enumerate(zip(bars, model_data['CHI'])):
            if pd.notna(val):
                height = bar.get_height()
                if height > 0:  # Only add text if bar has height
                    axes[0].text(bar.get_x() + bar.get_width()/2., height/2,
                                f'{val:.1f}', ha='center', va='center', fontsize=5, 
                                rotation=90, color='black')
    
    axes[0].set_ylabel('CHI Score', fontsize=12)
    axes[0].set_title('Calinski-Harabasz Index (Higher is Better)', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=min(num_models, 3))
    axes[0].grid(axis='y', linestyle='--', alpha=0.3)
    
    # --- Plot XB (raw) ---
    for i, (model_name, model_data) in enumerate(zip(model_names, all_models_data)):
        offset = (i - num_models/2 + 0.5) * bar_width
        bars = axes[1].bar(x + offset, model_data['XB'], bar_width, label=model_name, color=colors[i])
        
        # Add value labels inside bars horizontally
        for j, (bar, val) in enumerate(zip(bars, model_data['XB'])):
            if pd.notna(val) and np.isfinite(val):
                height = bar.get_height()
                if height > 0:  # Only add text if bar has height
                    axes[1].text(bar.get_x() + bar.get_width()/2., height/2,
                                f'{val:.3f}', ha='center', va='center', fontsize=5, 
                                rotation=90, color='black')
    
    axes[1].set_ylabel('XB Score', fontsize=12)
    axes[1].set_title('Xie–Beni Index (Lower is Better)', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=min(num_models, 3))
    axes[1].grid(axis='y', linestyle='--', alpha=0.3)
    
    # --- Plot AQI ---
    for i, (model_name, model_data) in enumerate(zip(model_names, all_models_data)):
        offset = (i - num_models/2 + 0.5) * bar_width
        bars = axes[2].bar(x + offset, model_data['AQI'], bar_width, label=model_name, color=colors[i])
        
        # Add value labels inside bars horizontally
        for j, (bar, val) in enumerate(zip(bars, model_data['AQI'])):
            if pd.notna(val):
                height = bar.get_height()
                if height > 0:  # Only add text if bar has height
                    axes[2].text(bar.get_x() + bar.get_width()/2., height/2,
                                f'{val:.1f}', ha='center', va='center', fontsize=5, 
                                rotation=90, color='black')
    
    axes[2].set_ylabel('AQI Score [0-100]', fontsize=12)
    axes[2].set_title('Alignment Quality Index (AQI) (Higher is Better)', fontsize=14, fontweight='bold')
    axes[2].set_ylim(0, 105)
    axes[2].legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=min(num_models, 3))
    axes[2].grid(axis='y', linestyle='--', alpha=0.3)
    
    # --- Common X-axis settings ---
    axes[2].set_xlabel("Category", fontsize=12)
    plt.xticks(x, valid_keys_for_plot, rotation=45, ha='right', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # --- Save figure ---
    plot_filename = os.path.join(save_dir, "multi_model_metrics_summary.png")
    try:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved multi-model metrics summary plot to {plot_filename}")
    except Exception as e:
        print(f"Error saving summary plot {plot_filename}: {e}")
    plt.show()
    
    return summary_df