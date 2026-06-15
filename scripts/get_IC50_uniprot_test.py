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
    try:
        response = requests.post(
            url, json={"query": query, "variables": {"ids": pdb_ids}}
        )
        if response.status_code == 200:
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