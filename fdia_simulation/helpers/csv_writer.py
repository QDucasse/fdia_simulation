# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 09:04:08 2019

@author: qde
"""
import csv
from datetime import datetime

class CSVWriter(object):
    '''
    Implements a helper to write the process noise best values found by the
    process noise finders.
    Parameters
    ----------
    filename: str
        Name of the file where the results will be writter.
    '''
    def __init__(self,filename = None):
        if filename is None:
            now = datetime.now()
            date_time = now.strftime("%d-%m-%Y_%H-%M")
            filename = './results/noise_finder_results-' + date_time + '.csv'
        self.filename = filename

    def write_row(self,model,q_value):
        '''
        Opens the file and write a row as: model, best value.
        Parameters
        ----------
        model: str
            Name of the filter model.

        q_value: str
            Best q value obtained by the process noise finder.

        Notes
        -----
        The file is closed thanks to the with encapsulation.
        '''
        with open(self.filename, mode = 'a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([model,q_value])

if __name__ == "__main__":
    writer = CSVWriter()
    writer.write_row('CA&1Radar','3200')
    writer.write_row('CA&2Radars','5')
    writer.write_row('CV&1Radar','300')
