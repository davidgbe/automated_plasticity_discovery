import csv
import numpy as np

def write_csv(out_file_path, data, delimiter=' '):
	with open(out_file_path, 'a', newline='\n') as csv_file:
		writer = csv.writer(csv_file, delimiter=delimiter)

		if type(data) is np.ndarray:
			if len(data.shape) > 1:
				for i in range(data.shape[0]):
					writer.writerow(data[i, :])
			else:
				writer.writerow(data)

		elif type(data) is list:
			if len(data) > 0 and (type(data[0]) is list or type(data[0]) is np.ndarray):
				for data_row in data:
					writer.writerow(list(data_row))
			else:
				writer.writerow(data)
		else:
			raise ValueError('Data is not of type list or ndarray!')
