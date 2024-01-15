import pandas as pd
import xlrd

print("Data load initiated, expect data load time of up to 3 minutes for all sheets.")
timeStart = pd.Timestamp.now()
# sheet_name = None: read all sheets
data = pd.read_excel(
    'data/ConsumptionExportFor2021AndJanary2022.xlsx', sheet_name="2021-02")
timeEnd = pd.Timestamp.now()
print("Data load time: ", timeEnd - timeStart)
print("Data load completed")
print("Data type: ", type(data))
if isinstance(data, dict):
    print("Data keys: ", data.keys())

elif isinstance(data, pd.DataFrame):
    print("Data shape: ", data.shape)
    print("Data columns: ", data.columns)
    print(data)

else:
    print("Data type not recognized")
    print("Data: ", data)
