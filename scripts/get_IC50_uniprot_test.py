"""
Adding Uniprot info to the csv file (test set)
Data Processing Pipeline for PDB-to-UniProt Cross-Referencing 

This script reads a dataset of protein-ligand binding affinities, calculates the 
negative log binding affinity (pIC50), queries the RCSB GraphQL API in batches 
to fetch mapped UniProt IDs for each PDB entry, and saves the integrated 
dataset to a CSV file.

We wanted to get the Uniprot ID bc the original dataset had it; but the prototype 
dataset didn't have these features, so we implemented it.
"""
import pandas as pd
import requests

# 1. Define file paths
input_file = "/home/bdldt_team001/DLDockingBenchSeminar/data/proto_test.csv"
output_file = ("/home/bdldt_team001/DLDockingBenchSeminar/data/proto_test_processed.csv")

print(f"Reading file from: {input_file}")
df = pd.read_csv(input_file)

# 2. Generate the pIC50 column (-1 * Log Binding Affinity)
print("Calculating pIC50 column...")
df["pIC50"] = -df["Log Binding Affinity"]

# 3. Setup the RCSB GraphQL API to query UniProt mappings
print("Extracting unique PDB IDs for mapping...")
# Ensure PDB IDs are strings, dropped NaNs, and unique
unique_pdb_ids = (
    df["PDBID"].dropna().astype(str).str.upper().unique().tolist()
)

url = "https://data.rcsb.org/graphql"
query = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    polymer_entities {
      rcsb_polymer_entity_container_identifiers {
        uniprot_ids
      }
    }
  }
}
"""


def fetch_batch(pdb_ids):
    """Fetches polymer data from the RCSB GraphQL API for a specific batch of PDB IDs.

    Args:
        pdb_ids (List[str]): A list of uppercase PDB IDs to query.

    Returns:
        List[Dict[str, Any]]: A list of entry dictionaries retrieved from the API response.
                              Returns an empty list if the request fails.
    """
    try:
        response = requests.post(
            url, json={"query": query, "variables": {"ids": pdb_ids}}
        )
        if response.status_code == 200:
            # If the connection is successful(200), then retrieve the data
            return response.json().get("data", {}).get("entries", [])
    except Exception as e:
        print(f"Error fetching batch: {e}")
    return []


# Fetch data in chunks of 100 to avoid overloading API payload constraints
batch_size = 100
pdb_to_uniprot = {}

total_batches = -(-len(unique_pdb_ids) // batch_size)
print(
    f"Starting UniProt cross-referencing for {len(unique_pdb_ids)} unique PDB IDs across {total_batches} batches..."
)

for i in range(0, len(unique_pdb_ids), batch_size):
    batch = unique_pdb_ids[i : i + batch_size]
    print(f"Processing batch {i//batch_size + 1}/{total_batches}...")

    entries = fetch_batch(batch)

    if entries:
        for entry in entries:
            pdb_id = entry.get("rcsb_id")
            if not pdb_id:
                continue

            uniprot_set = set()
            polymer_entities = entry.get("polymer_entities") or []

            for entity in polymer_entities:
                identifiers = (
                    entity.get("rcsb_polymer_entity_container_identifiers")
                    or {}
                )
                uniprot_ids = identifiers.get("uniprot_ids")
                if uniprot_ids:
                    uniprot_set.update(uniprot_ids)

            # Map using lowercase keys to match your CSV file's casing ('10gs')
            pdb_to_uniprot[pdb_id.lower()] = (
                ", ".join(uniprot_set) if uniprot_set else "None"
            )

# 4. Map the fetched UniProt IDs back to your original DataFrame
print("Mapping UniProt IDs back to the dataset...")
df["UniProtID"] = df["PDBID"].astype(str).str.lower().map(pdb_to_uniprot)

# Default unmatched entries to 'None'
df["UniProtID"] = df["UniProtID"].fillna("None")

# 5. Save the newly structured dataframe
df.to_csv(output_file, index=False)
print(f"\nProcessing complete! Successfully saved to: {output_file}")
print(df[["PDBID", "Log Binding Affinity", "pIC50", "UniProtID"]].head())