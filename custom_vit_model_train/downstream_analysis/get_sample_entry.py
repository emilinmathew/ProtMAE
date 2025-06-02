import json
import os

# Define the path to the DSSP annotations file
dssp_file_path = 'proteinnet_data/dssp_annotations.json'

# Check if the file exists
if not os.path.exists(dssp_file_path):
    print(f"Error: DSSP annotations file not found at {dssp_file_path}")
    print("Please run proteinnet_download.py first to download and extract the data.")
else:
    print(f"Loading DSSP annotations from {dssp_file_path}...")
    try:
        # Load the JSON data
        with open(dssp_file_path, 'r') as f:
            dssp_data = json.load(f)

        print(f"Successfully loaded data. Type of data: {type(dssp_data)}")

        # Inspect the data structure and print samples
        if isinstance(dssp_data, dict):
            print(f"Number of entries (proteins): {len(dssp_data)}")
            print("Keys (sample protein IDs):")
            # Print first 5 keys
            for i, key in enumerate(list(dssp_data.keys())[:5]):
                print(f"- {key}")

            print("\nSample annotation structure for the first protein:")
            first_key = list(dssp_data.keys())[0]
            print(f"Annotation for '{first_key}': {dssp_data[first_key]}")

            # If the value is a list of annotations (one per residue)
            if isinstance(dssp_data[first_key], list):
                 print(f"Annotation for '{first_key}' is a list (likely one entry per residue). Length: {len(dssp_data[first_key])}")
                 # Print first 10 annotations if it's a list of many items
                 if len(dssp_data[first_key]) > 1:
                     print(f"First 10 annotation entries for '{first_key}': {dssp_data[first_key][:10]}")


        elif isinstance(dssp_data, list):
            print(f"Number of entries: {len(dssp_data)}")
            print("\nFirst 5 annotation entries:")
            for i, entry in enumerate(dssp_data[:5]):
                print(f"Entry {i+1}: {entry}")

        else:
            print("Data is neither a dictionary nor a list. Cannot display sample structure.")

    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {dssp_file_path}. The file might be corrupted.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
