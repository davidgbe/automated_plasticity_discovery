import csv
import re
import pandas as pd

class InFile(object):
    def __init__(self, infile):
        self.line_count = 0
        self.infile = open(infile)

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

    def read(self, *args, **kwargs):
        return self.__next__()
    
    def handle_stop(self):
        self.infile.close()
        raise StopIteration()
        
    def next(self):
        try:
            line: str = self.infile.readline()
        except Exception as e:
            print(e)
            self.handle_stop()
        
        if line == '':
            self.handle_stop()
            
        if self.line_count == 0:
            line = line[line.find('"') + 1:line.rfind('"')]
            line = re.sub(' ', '', line)
            line += '\n'
        else:
            line = re.sub(' ', ',', line.strip()) + '\n'

        self.line_count += 1

        return line
    
def read_csv(file_path, read_header=True):
    columns = None
    data = []

    file = InFile(file_path)

    for row_idx, row in enumerate(csv.reader(file, delimiter=',')):
        if row_idx == 0:
            columns = row
        else:
            data.append([float(num) for num in row])
    if read_header:
        df = pd.DataFrame(data=data, columns=columns[:-1])
    else:
        df = pd.DataFrame(data=data)
    
    return df