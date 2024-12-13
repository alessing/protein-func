import pandas as pd
import matplotlib.pyplot as plt

# Define file paths and column names
RAW_DATA = "data/raw_data"
PLOT_DIR = "data/plots"
PROCESSED_DATA = "data/processed_data"
COLUMNS = [
    'DB',
    'DB_Object_ID',
    'DB_Object_Symbol',
    'Qualifier',
    'GO_ID',
    'DB_Reference',
    'Evidence_Code',
    'With_From',
    'Aspect',
    'DB_Object_Name',
    'DB_Object_Synonym',
    'DB_Object_Type',
    'Taxon',
    'Date',
    'Assigned_By',
    'Annotation_Extension',
    'Gene_Product_Form_ID'
]

def num_functions_per_protein(lengths_df, max_x, file_path):
    """
    Create and save a histogram of the number of GO IDs per protein.

    :param lengths_df: Series containing the number of GO IDs per protein
    :param max_x: Maximum x-value for the histogram
    :param file_path: Path to save the histogram image
    """
    plt.figure(figsize=(10, 6))
    plt.hist(lengths_df, bins=30, edgecolor='black', color='skyblue', alpha=0.7, range=(0, max_x))
    plt.xlabel('Number of GO IDs')
    plt.ylabel('Number of Proteins')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.axvline(lengths_df.mean(), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {lengths_df.mean():.1f}')
    plt.axvline(lengths_df.median(), color='green', linestyle='dashed', linewidth=1, label=f'Median: {lengths_df.median():.1f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def main():
    # Load the raw data file, filtering out comments and setting column names
    data = pd.read_csv(f"{RAW_DATA}/goa_human.gaf", sep="\t", names=COLUMNS, comment="!", na_values=[""], dtype=str)

    # Filter data to only include molecular function annotations for proteins
    data = data[data["Aspect"] == "F"]  # Only keep annotations detailing molecular function
    data = data[data["DB_Object_Type"] == "protein"]  #  Only keep annotations detailing proteins

    # Print the count of each qualifier type
    for qual in set(data["Qualifier"]):
        print(f'{qual}: {len(data[data["Qualifier"] == qual])}')
    
    # Map qualifiers to indices for easier processing
    data["Qualifier_Idx"] = data["Qualifier"].apply(lambda x: 0 if x == "enables" else (1 if x == "contributes_to" else 2))

    # Remove duplicate annotations based on protein ID and GO ID
    data = data.drop_duplicates(subset=["DB_Object_ID", "GO_ID"], keep='first')

    # Group data by protein ID, aggregating lists of qualifier indices and GO IDs
    protein_df = data.groupby("DB_Object_ID").agg({"Qualifier_Idx": list, "GO_ID": list}).reset_index()

    # Create a mapping from GO IDs to indices
    go_id_set = set()
    for go_id_list in protein_df["GO_ID"]:
        for go_id in go_id_list:
            go_id_set.add(go_id)
    go_id_to_idx = {go_id: idx for idx, go_id in enumerate(go_id_set)}

    # Convert GO IDs to indices using the mapping
    protein_df["GO_Idx"] = protein_df['GO_ID'].apply(lambda x: [go_id_to_idx[go_id] for go_id in x])

    # Calculate the number of annotations per protein and plot histograms
    number_annotations = protein_df['GO_ID'].apply(len)
    max_annotations = max(number_annotations)
    num_functions_per_protein(number_annotations, max_annotations, f"{PLOT_DIR}/num_functions_per_protein.jpg")
    num_functions_per_protein(number_annotations, 100, f"{PLOT_DIR}/small_num_functions_per_protein.jpg")

    # Save the processed data to a CSV file
    protein_df.to_csv(f"{PROCESSED_DATA}/protein_functions.csv")

if __name__ == "__main__":
    main()
