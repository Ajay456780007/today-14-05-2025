import pandas as pd
import numpy as np
# === Load BEDGRAPH file ===
chip_path = "dataset/dataset1/chip-seq_file/Galaxy10-[bigWigToBedGraph on data 8].bedgraph"
chip_df = pd.read_csv(
    chip_path,
    sep="\t",
    header=None,
    names=["chrom", "start", "end", "signal"],
    dtype={"chrom": str}  # make sure chromosome is treated as string
)

# === Keep only chromosomes "1" to "10" ===
chroms_to_keep = [str(i) for i in range(1, 11)]
filtered_chip_df = chip_df[chip_df["chrom"].isin(chroms_to_keep)]

# === Save to CSV ===
filtered_chip_df.to_csv("dataset/dataset1/chip-seq_file/chipseq_chr1_to_10.csv", index=False)

print("Saved chromosomes 1 to 10 to 'chipseq_chr1_to_10.csv'")


import os
import gzip
import pandas as pd
import numpy as np
from urllib.parse import unquote
from Bio import SeqIO
dna_dir = "dataset/dataset1/dna_chromosomes"
gff3_dir = "dataset/dataset1/gff3_files"
#
# # === Collect all FASTA and GFF3 files ===
fasta_files = sorted([
     os.path.join(dna_dir, f) for f in os.listdir(dna_dir)
     if f.lower().endswith(".fa.gz")
 ])
gff3_files = sorted([
     os.path.join(gff3_dir, f) for f in os.listdir(gff3_dir)
     if f.lower().endswith(".gff3")
])
#
#
# === Parse GFF3 attributes ===
def parse_attributes(attr_str):
    attr_dict = {}
    for pair in attr_str.strip().split(";"):
        if "=" in pair:
            key, value = pair.split("=", 1)
            attr_dict[key.strip()] = unquote(value.strip())
    return attr_dict
#
#
# # === Function to parse GFF3 and extract gene entries for a given chromosome ===
def parse_gff3(gff3_file, chrom_id):
    genes = []
    with open(gff3_file, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            if parts[2] != "gene":
                continue
            if parts[0] != chrom_id:
                continue  # skip if this line refers to a different chromosome

            start = int(parts[3]) - 1  # Convert to 0-based index
            end = int(parts[4])
            strand = parts[6]
            attrs = parse_attributes(parts[8])
            gene_id = attrs.get("ID", "NA")

            genes.append((start, end, strand, gene_id))
    return genes

# === Extract gene sequences ===
gene_sequences = []

for fasta_path, gff_path in zip(fasta_files, gff3_files):
    print(f"Processing: {os.path.basename(fasta_path)} with {os.path.basename(gff_path)}")

    # Read the chromosome sequence (support gzipped or uncompressed)
    if fasta_path.endswith(".gz"):
         with gzip.open(fasta_path, "rt", encoding="utf-8") as f:
             record = next(SeqIO.parse(f, "fasta"))
    else:
         with open(fasta_path, "r", encoding="utf-8") as f:
            record = next(SeqIO.parse(f, "fasta"))

    chrom_seq = record.seq
    chrom_id = record.id

    # Parse gene entries from the GFF3 file
    genes = parse_gff3(gff_path, chrom_id)

    for start, end, strand, gene_id in genes:
        # Boundary check to avoid errors
        if start < 0 or end > len(chrom_seq):
            continue
#
        gene_seq = chrom_seq[start:end]
        if strand == "-":
            gene_seq = gene_seq.reverse_complement()

        gene_sequences.append({
            "gene_id": gene_id,
            "chrom": chrom_id,
            "start": start,
            "end": end,
            "strand": strand,
            "sequence": str(gene_seq)
        })
# === Convert to DataFrame ===
df_genes = pd.DataFrame(gene_sequences)

# === Preview first few entries ===
print(df_genes.head())
geo_path = "dataset/dataset1/geo_files/genes_to_alias_ids.tsv"
df = pd.read_csv(geo_path, sep='\t')
alias_path = "dataset/dataset1/expression level TPM/abundance.tsv"
df_alias = pd.read_csv(alias_path, sep='\t')
df.rename(columns={"Zm00001eb000010": "id1", "B73 Zm00001eb.1": "id2", "Zm00001d027230": "gene_alias_id"})

import pandas as pd
#
# # Fix column names in 'df'
df.columns = ['id1', 'id2', 'gene_alias_id', 'AGPv4_Zm00001d.2']
#
# # Step 1: Clean the 'gene_id' column in df_genes
df_genes['gene_id_clean'] = df_genes['gene_id'].str.replace('gene:', '', regex=False)
#
# # Step 2: Create a mapping from 'id1' to 'gene_alias_id'
id_to_alias = df.set_index('id1')['gene_alias_id'].to_dict()
#
# # Step 3: Map gene_alias_id to df_genes
df_genes['alias_id'] = df_genes['gene_id_clean'].map(id_to_alias)
#
# # Step 4: Clean up the temporary column
df_genes.drop(columns=['gene_id_clean'], inplace=True)
#
# # Check the results
print(df_genes[['gene_id', 'alias_id']].head(50))
#
# # Drop rows where alias_id is NaN
df_genes = df_genes.dropna(subset=['alias_id'])
#
# # Reset index if needed
df_genes = df_genes.reset_index(drop=True)
#
# # Optional: check that it's gone
print(df_genes['alias_id'].isna().sum())
#

#
# # Step 1: Clean up df_alias to remove transcript suffix
df_alias['clean_id'] = df_alias['target_id'].str.replace(r'_T\d+$', '', regex=True)
#
# # Step 2: Group by clean_id and sum or average TPM if needed (in case multiple transcripts per gene)
# # Here we'll sum TPMs for all isoforms of a gene
tpm_by_gene = df_alias.groupby('clean_id')['tpm'].sum().reset_index()
#
# # Step 3: Merge df_genes with this TPM info using alias_id == clean_id
df_genes = df_genes.merge(tpm_by_gene, how='left', left_on='alias_id', right_on='clean_id')
#
# # Step 4: Rename the column to tpm_value and drop clean_id
df_genes = df_genes.rename(columns={'tpm': 'tpm_value'}).drop(columns=['clean_id'])
#
# # Done
print(df_genes.head())
#
# # Mapping dictionary
base_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
#
#
# # Function to encode a DNA sequence string
def encode_sequence(seq):
     return [base_map.get(base, -1) for base in seq.upper()]  # -1 for unknown bases like N
#
#
# # Apply to each row in df_genes['sequence']
df_genes['encoded_sequence'] = df_genes['sequence'].apply(encode_sequence)
#
# # Done
print(df_genes[['sequence', 'encoded_sequence']].head())
#
df_genes.drop("sequence", axis=1, inplace=True)
#

max_len = df_genes['encoded_sequence'].apply(len).max()
print("Maximum encoded sequence length:", max_len)
#
# # Step 1: Store lengths of all encoded sequences
sequence_lengths = df_genes['encoded_sequence'].apply(len)
#
# # Step 2: Count how many are greater than 50,000
num_greater_than_40000 = (sequence_lengths > 1000).sum()
#
# # Output results
print("Total sequences:", len(sequence_lengths))
print("Sequences > 50,000 bases:", num_greater_than_40000)
#
# Fixed length
FIXED_LEN = 3000
PAD_VALUE = 0  # A = 0

#
def pad_or_truncate(seq):
    if len(seq) > FIXED_LEN:
        return seq[:FIXED_LEN]
    else:
        return seq + [PAD_VALUE] * (FIXED_LEN - len(seq))  # pad
#
#
# # Apply to each sequence
df_genes['encoded_50k'] = df_genes['encoded_sequence'].apply(pad_or_truncate)

# Check shape of one example
print(len(df_genes['encoded_50k'].iloc[0]))
#
print(df_genes.head())

df_genes['strand'] = df_genes['strand'].map({'+': 1, '-': 0})

df_genes.rename(columns={"encoded_50k": "sequence"}, inplace=True)

df_genes["sequence"]
#

#
#
def fix_sequence(seq):
    return [4 if val == -1 else val for val in seq]
#

df_genes['sequence'] = df_genes['sequence'].apply(fix_sequence)
x = np.array(df_genes['sequence'].tolist(), dtype=np.uint8)



y = df_genes["tpm_value"]

print("this is x shape:", x.shape)
print("this is y shape:", y.shape)

print(df_genes.head())

df_genes["encoded_sequence"].apply(len)

df_genes.drop(columns={"encoded_sequence"}, inplace=True)

print(df_genes.head())
df_genes.to_csv("dataset/dataset1/data.csv", index=False)

import pandas as pd
data=pd.read_csv("dataset/dataset1/chip-seq_file/chipseq_chr1_to_10.csv")
print(data.head())


import pandas as pd
import pyranges as pr

print(" Step 1: Loading ChIP-Seq bedgraph file...")
chip_df = pd.read_csv("dataset/dataset1/chip-seq_file/chipseq_chr1_to_10.csv")
print(f" chip_df loaded: {chip_df.shape[0]} rows")

print(" Step 2: Loading gene dataset...")
gene_df = pd.read_csv("dataset/dataset1/data.csv")  # Make sure this file contains a 'gene_id' column
print(f" gene_df loaded: {gene_df.shape[0]} rows")

# === Ensure chromosome columns are strings ===
print("Step 3: Preparing columns for PyRanges...")
chip_df["Chromosome"] = chip_df["chrom"].astype(str)
chip_df["Start"] = chip_df["start"]
chip_df["End"] = chip_df["end"]
chip_df["Signal"] = chip_df["signal"]

gene_df["Chromosome"] = gene_df["chrom"].astype(str)
gene_df["Start"] = gene_df["start"]
gene_df["End"] = gene_df["end"]

# Use a unique gene identifier
gene_df["GeneID"] = gene_df["gene_id"]  # Adjust if your column is named differently
print("Columns renamed and cast to correct types.")

# === Convert to PyRanges objects ===
print(" Step 4: Converting DataFrames to PyRanges...")
chip_ranges = pr.PyRanges(chip_df[["Chromosome", "Start", "End", "Signal"]])
gene_ranges = pr.PyRanges(gene_df[["Chromosome", "Start", "End", "GeneID"]])
print(" PyRanges objects created.")

# === Perform intersection ===
print(" Step 5: Performing genomic join (intersection)...")
joined = gene_ranges.join(chip_ranges, strandedness=False)
print(f" Join completed: {joined.df.shape[0]} overlapping records found")

# === Group by GeneID and calculate average ChIP signal ===
print(" Step 6: Grouping and averaging ChIP signal for each gene...")
grouped = joined.df.groupby("GeneID", observed=True)["Signal"].mean().reset_index()
grouped.rename(columns={"Signal": "avg_chip_signal"}, inplace=True)
print(f" Averaging complete: {grouped.shape[0]} genes with signal")

# === Merge back with original gene_df ===
print(" Step 7: Merging signal data back into gene dataset...")
result = pd.merge(gene_df, grouped, on="GeneID", how="left")
result["avg_chip_signal"] = result["avg_chip_signal"].fillna(0)
print(" Merge complete.")

# === Save the combined dataset ===
print(" Saving final result to 'gene_dataset_with_chip.csv'...")
result.to_csv("dataset/dataset1/gene_dataset_with_chip.csv", index=False)
print("🎉 Done! Combined dataset saved as 'gene_dataset_with_chip.csv'")

import pandas as pd
df_output=pd.read_csv("dataset/dataset1/gene_dataset_with_chip.csv")


df_output.head()

import pandas as pd

# Load the full CSV file
df = pd.read_csv("dataset/dataset1/gene_dataset_with_chip.csv")

# Drop redundant or duplicate columns
df_cleaned = df.drop(columns=["Chromosome", "Start", "End", "GeneID"])

# Optional: rename 'gene_id' if needed (e.g., remove 'gene:' prefix)
df_cleaned['gene_id'] = df_cleaned['gene_id'].str.replace('gene:', '', regex=False)

# Display a preview
print(df_cleaned.head())

print(df_cleaned.head())

import json
df_cleaned["sequence"] = df_cleaned["sequence"].apply(json.loads)

print(df_cleaned["sequence"].apply(len))
import numpy as np
import ast
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter


# Step 2: Normalize ChIP-seq signal
df_cleaned["avg_chip_signal"] = df_cleaned["avg_chip_signal"] / df_cleaned["avg_chip_signal"].max()

# Step 3: Prepare X: sequence + chip signal
X_seq = np.array(df_cleaned["sequence"].tolist())  # shape: (N, 3000)
chip = df_cleaned["avg_chip_signal"].values.reshape(-1, 1)  # shape: (N, 1)
chip_tiled = np.repeat(chip, X_seq.shape[1], axis=1)  # shape: (N, 3000)
X_combined = np.concatenate([X_seq, chip_tiled], axis=1)  # shape: (N, 6000)

# Step 4: Categorize TPM values
tpm_values = df_cleaned["tpm_value"].values
mean_tpm = np.mean(tpm_values)
low_threshold = mean_tpm / 2
high_threshold = mean_tpm * 1.5

def categorize_tpm(tpm, low, high):
    if tpm < low:
        return 0  # Low expression
    elif tpm < high:
        return 1  # Medium expression
    else:
        return 2  # High expression

categorized_labels = np.array([categorize_tpm(t, low_threshold, high_threshold) for t in tpm_values])
categorized_labels = categorized_labels.astype(np.int32)

# Step 5: Stratified Train-Test Split
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in splitter.split(X_combined, categorized_labels):
    x_train, x_test = X_combined[train_idx], X_combined[test_idx]
    y_train, y_test = categorized_labels[train_idx], categorized_labels[test_idx]

np.save("train_test_data_for_copy/x_train_zea.npy", x_train)
np.save("train_test_data_for_copy/x_test_zea.npy", x_test)
np.save("train_test_data_for_copy/y_train_zea.npy", y_train)
np.save("train_test_data_for_copy/y_test_zea.npy", y_test)

#  Print class distributions
print("Train distribution:", Counter(y_train))
print("Test distribution:", Counter(y_test))
print(" Data saved. Shapes → x_train:", x_train.shape, ", y_train:", y_train.shape)




# import tensorflow as tf
# from tensorflow.keras import layers
#
# def get_positional_encoding(seq_len, model_dim):
#     import numpy as np
#     angle_rads = np.arange(seq_len)[:, np.newaxis] / np.power(
#         10000, (2 * (np.arange(model_dim)[np.newaxis, :] // 2)) / np.float32(model_dim)
#     )
#     angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
#     angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
#     return tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)
#
# class TransformerBlock(layers.Layer):
#     def __init__(self, model_dim, heads, ff_dim, rate=0.1):
#         super().__init__()
#         self.att = layers.MultiHeadAttention(num_heads=heads, key_dim=model_dim)
#         self.ffn = tf.keras.Sequential([
#             layers.Dense(ff_dim, activation='gelu'),
#             layers.Dense(model_dim),
#         ])
#         self.norm1 = layers.LayerNormalization(epsilon=1e-6)
#         self.norm2 = layers.LayerNormalization(epsilon=1e-6)
#         self.dropout1 = layers.Dropout(rate)
#         self.dropout2 = layers.Dropout(rate)
#
#     def call(self, x, training=False):
#         attn_output = self.att(x, x)
#         out1 = self.norm1(x + self.dropout1(attn_output, training=training))
#         ffn_output = self.ffn(out1)
#         return self.norm2(out1 + self.dropout2(ffn_output, training=training))
#
# def build_diffusion_transformer(seq_len=6000, vocab_size=5, model_dim=64, num_heads=4, ff_dim=128, depth=2):
#     inputs = layers.Input(shape=(seq_len,), dtype='int32', name='dna_sequence')
#     x = layers.Embedding(input_dim=vocab_size, output_dim=model_dim)(inputs)
#     pos_encoding = get_positional_encoding(seq_len, model_dim)
#     x = x + pos_encoding[:, :seq_len, :]
#     for _ in range(depth):
#         x = TransformerBlock(model_dim, num_heads, ff_dim)(x)
#     x = layers.GlobalAveragePooling1D()(x)
#     model = tf.keras.Model(inputs, x, name="DiffusionTransformer")
#     return model
#
#
# class HGNNConv(layers.Layer):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.weight = self.add_weight(shape=(input_dim, output_dim),
#                                       initializer='glorot_uniform',
#                                       trainable=True)
#         self.bias = self.add_weight(shape=(output_dim,),
#                                     initializer='zeros',
#                                     trainable=True)
#
#     def call(self, x, G):
#         # x: (batch, nodes, features)
#         # G: (batch, nodes, nodes) adjacency matrix or incidence matrix
#         x = tf.matmul(x, self.weight) + self.bias  # linear transformation
#         x = tf.matmul(G, x)                        # message passing on graph
#         return tf.nn.relu(x)
#
# class HGNNEmbedding(tf.keras.Model):
#     def __init__(self, input_dim, hidden_dim, dropout_rate=0.5):
#         super().__init__()
#         self.hgc1 = HGNNConv(input_dim, hidden_dim)
#         self.hgc2 = HGNNConv(hidden_dim, hidden_dim)
#         self.dropout_rate = dropout_rate
#
#     def call(self, x, G, training=False):
#         x = self.hgc1(x, G)
#         if training:
#             x = tf.nn.dropout(x, rate=self.dropout_rate)
#         x = self.hgc2(x, G)
#         return tf.reduce_mean(x, axis=1)  # global pooling over nodes
#
#
# def build_hybrid_attention_fusion(input_dim_seq, input_dim_hg, fusion_dim=128, num_classes=3):
#     # Inputs
#     input_seq_feat = layers.Input(shape=(input_dim_seq,), name='seq_features')
#     input_hg_feat = layers.Input(shape=(input_dim_hg,), name='hg_features')
#
#     # Concatenate features
#     x = layers.Concatenate()([input_seq_feat, input_hg_feat])
#     x = layers.Dense(fusion_dim, activation='relu')(x)
#     x = layers.Dropout(0.2)(x)
#     x = layers.Dense(fusion_dim//2, activation='relu')(x)
#     x = layers.Dropout(0.2)(x)
#     output = layers.Dense(num_classes, activation='softmax')(x)
#
#     model = tf.keras.Model(inputs=[input_seq_feat, input_hg_feat], outputs=output, name='HybridAttentionFusion')
#     return model
# # Inputs
# dna_input = layers.Input(shape=(6000,), dtype='int32', name='dna_input')
# hg_input = layers.Input(shape=(X_hg.shape[1], X_hg.shape[2]), dtype='float32', name='hg_input')
# hg_adj_input = layers.Input(shape=(X_hg.shape[1], X_hg.shape[1]), dtype='float32', name='hg_adj_input')  # hypergraph adjacency
#
# # Diffusion Transformer output
# diffusion_model = build_diffusion_transformer()
# seq_features = diffusion_model(dna_input)  # (batch, model_dim)
#
# # HGNN output
# hgnn_model = HGNNEmbedding(input_dim=X_hg.shape[2], hidden_dim=64)
# hg_features = hgnn_model(hg_input, hg_adj_input)  # (batch, 64)
#
# # Hybrid Attention Fusion
# fusion_model = build_hybrid_attention_fusion(seq_features.shape[-1], hg_features.shape[-1])
# output = fusion_model([seq_features, hg_features])
#
# # Final combined model
# final_model = tf.keras.Model(inputs=[dna_input, hg_input, hg_adj_input], outputs=output)
# final_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# final_model.summary()
#
# # x_seq_train: (num_samples, 3000)
# # x_hg_train: (num_samples, num_nodes, feature_dim)
# # x_hg_adj_train: (num_samples, num_nodes, num_nodes)
# # y_train: (num_samples, )
#
# final_model.fit(
#     [x_seq_train, x_hg_train, x_hg_adj_train],
#     y_train,
#     validation_data=([x_seq_test, x_hg_test, x_hg_adj_test], y_test),
#     epochs=10,
#     batch_size=32
# )
#
#
