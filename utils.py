import pandas as pd
import json
import os
import weaviate
from tqdm.notebook import tqdm


def parse_embeddings(texts, model, batch_size=1024):
    NUM_BATCH = len(texts) // batch_size
    if len(texts) % batch_size != 0:
        NUM_BATCH += 1

    results = []
    for batch_idx in tqdm(range(NUM_BATCH)):
        results.extend(model.encode(texts[batch_idx * batch_size: (batch_idx + 1) * batch_size]).tolist())

    return results


def convert_json_to_dataframe(json_file_path):
    """Converts a JSON file into a Pandas DataFrame with columns as key-value pairs.

    Args:
        json_file_path: The path to the JSON file.

    Returns:
        A Pandas DataFrame containing key-value pairs from the JSON file.
    """

    with open(json_file_path, "r") as f:
        json_data = json.load(f)

    # Flatten the nested JSON data to key-value pairs.
    flattened_data = flatten_json(json_data)

    # Create a DataFrame from the flattened data.
    df = pd.DataFrame([flattened_data])

    return df


def flatten_json(json_data, parent_key='', sep='_'):
    """Flattens a nested JSON object to key-value pairs.

    Args:
        json_data: The nested JSON object.
        parent_key: The parent key (used for recursion).
        sep: The separator between keys.

    Returns:
        A flattened dictionary of key-value pairs.
    """
    items = []
    for k, v in json_data.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_json(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def convert_all_jsons_to_dataframe(folder_path):
    """Converts all JSON files in a folder path into a Pandas DataFrame.

    Args:
        folder_path: The path to the folder containing the JSON files.

    Returns:
        A Pandas DataFrame containing key-value pairs from all JSON files in the folder path.
    """

    # Get a list of all JSON files in the folder path.
    json_file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".json")]

    # Iterate over the list of JSON files and convert each file into a DataFrame.
    dataframes = []
    for json_file_path in json_file_paths:
        dataframe = convert_json_to_dataframe(json_file_path)
        dataframes.append(dataframe)

    # Concatenate the list of DataFrames into a single DataFrame.
    df = pd.concat(dataframes, ignore_index=True)

    return df


def parse_text(properties):
    # print(properties.keys())
    # print(properties["Label"], type(properties["Label"]))
    str_in = ''
    for k in sorted(properties.keys()):
        v = properties[k]
        if isinstance(v, str):
            str_in += v + ' '
    str_in = str_in.lower().strip()  # remove trailing whitespace
    str_in = f"{SCHEMA_NAME} {str_in}"
    return str_in


def weaviate_client():
    client = weaviate.Client(
        embedded_options=weaviate.embedded.EmbeddedOptions(),
        additional_headers={
            # 'X-OpenAI-Api-Key': 'YOUR-OPENAI-API-KEY'  # Replace w/ your OPENAI API key,
            # "X-Huggingface-Api-Key": "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"
        }
    )
    return client
