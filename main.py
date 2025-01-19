import NormalizeInput as NI
import InputCSV as ICSV

dataMatrix1, stockNames = ICSV.load_csv_to_matrix("stock_prices_sample.csv")
dataMatrix, stats = NI.CleanUp(dataMatrix1)
ICSV.print_matrix_and_names(NI.round_data_matrix(dataMatrix), stockNames)
ICSV.print_matrix_and_names(ICSV.load_csv_to_matrix("stock_prices_sample.csv")[0], ICSV.load_csv_to_matrix("stock_prices_sample.csv")[1])

print("hello world")