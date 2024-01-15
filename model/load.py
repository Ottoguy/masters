import pandas as pd
import xlrd

print("Data load initiated, expect data load time of up to 3 minutes.")
timeStart = pd.Timestamp.now()
# sheet_name = None: read all sheets
data = pd.read_excel(
    'data/ConsumptionExportFor2021AndJanary2022.xlsx', sheet_name=None, nrows=float('inf'))
timeEnd = pd.Timestamp.now()
print("Data load time: ", timeEnd - timeStart)
print("Data load completed")
print(data)
