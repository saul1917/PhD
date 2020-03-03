# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

from datetime import datetime
from collections import defaultdict
import threading
import time
import logging
import os
import sys
from pandas import DataFrame
from collections import defaultdict


class TrainLog:
    """Saves training logs in Pandas csvs"""

    INCREMENTAL_UPDATE_TIME = 300

    def __init__(self, directory, name):
        self.log_file_path = "{}/{}.csv".format(directory, name)
        self._log = defaultdict(dict)
        self._log_lock = threading.RLock()
        self._last_update_time = time.time() - self.INCREMENTAL_UPDATE_TIME

    def record_single(self, step, column, value):
        self._record(step, {column: value})

    def record(self, step, col_val_dict):
        self._record(step, col_val_dict)

    def save(self):
        df = self._as_dataframe()
        #df.to_msgpack(self.log_file_path, compress='zlib')
        df.to_csv(self.log_file_path)

    def _record(self, step, col_val_dict):
        with self._log_lock:
            self._log[step].update(col_val_dict)
            #if time.time() - self._last_update_time >= self.INCREMENTAL_UPDATE_TIME:
            self._last_update_time = time.time()
            self.save()

    def _as_dataframe(self):
        with self._log_lock:
            return DataFrame.from_dict(self._log, orient='index')


class RunContext:
    """Creates directories and files for the run"""
    def __init__(self, logging):
        dateInfo = "{date:%Y-%m-%d_%H_%M_%S}".format(date=datetime.now())
        #runner_name = os.path.basename(runner_file).split(".")[0]
        self.result_dir = ("../{root}/" + dateInfo + "/").format(root = 'logs')
        #transient dir contains the checkpoints, information logs and training logs
        self.transient_dir = self.result_dir + "logs/"
        os.makedirs(self.result_dir)
        os.makedirs(self.transient_dir)
        logging.basicConfig(filename=self.transient_dir + "log_" + dateInfo + ".txt", level=logging.INFO, format='%(message)s')
        self.LOG = logging.getLogger('main')
        self.init_logger()
        #creating log in log dir
        self.LOG.info("Creating directories for execution: ")
        self.LOG.info(self.result_dir)
        self.LOG.info(self.transient_dir)


    def init_logger(self):
        """
        Sets logging details
        :return:
        """
        #self.LOG.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.LOG.addHandler(handler)

    def get_logger(self):
        return self.LOG

    def create_train_log(self, name):
        return TrainLog(self.transient_dir, name)

    def create_results_all_log(self,name, directory = "../logs/summary"):
        return TrainLog(directory, name)
