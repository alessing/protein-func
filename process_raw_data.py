import pandas as pd
import matplotlib.pyplot as plt

RAW_DATA = "data/raw_data"
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
    data = pd.read_csv(f"{RAW_DATA}/goa_human.gaf", sep="\t", names=COLUMNS, comment="!", na_values=[""], dtype=str)
    data = data[data["Aspect"] == "F"]  # Only keep annotations detailing molecular function
    data = data[data["DB_Object_Type"] == "protein"]  #  Only keep annotations detailing proteins
    protein_df = data.groupby("DB_Object_ID").agg({"Qualifier": list, "GO_ID": list})
    
    number_annotations = protein_df['GO_ID'].apply(len)
    max_annotations = max(number_annotations)
    num_functions_per_protein(number_annotations, max_annotations, "data/plots/num_functions_per_protein.jpg")
    num_functions_per_protein(number_annotations, 100, "data/plots/small_num_functions_per_protein.jpg")

if __name__ == "__main__":
    main()
