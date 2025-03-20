import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
code_data = pd.read_json("https://datasets-server.huggingface.co/rows?dataset=nvidia%2FLlama-Nemotron-Post-Training-Dataset-v1&config=SFT&split=code&offset=0", lines=True)
code_data.to_csv("code_data.csv")
