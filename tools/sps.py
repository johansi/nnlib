import snap7
import snap7.util
from .helper import *


class SPS_CONNECT:

    def __init__(self, ip, db, rack, slot, targets_bool, targets_real):
        self.db = None

        self.targets_bool = targets_bool
        self.targets_real = targets_real
        self.sps_values = None
        self.db_set = db
        self.plc = snap7.client.Client()
        try:
            self.plc.connect(ip,rack,slot)
            if self.plc.get_connected():
                printing("CONNECT TO SPS "+ ip, print_types.INFO)                
                #pdb.set_trace()
                self.read_db()
                self.read()
                self.set_off_state()
            else:
                printing("CONNECTION TO SPS FAILED", print_types.WARNING)
                
        except:
            printing("CONNECTION TO SPS FAILED", print_types.WARNING)


    def set_sps_value(self,key,value):
        if self.sps_values is not None:
            self.sps_values[key] = value

    def set_off_state(self):
        if self.sps_values is not None:
            for key in self.targets_bool.keys():
                self.sps_values[key] = self.targets_bool[key][2]
            for key in self.targets_real.keys():
                self.sps_values[key] = self.targets_real[key][2]
            self.write()

    def set_ready_state(self, write=True):
        if self.sps_values is not None:
            for key in self.targets_bool.keys():
                self.sps_values[key] = self.targets_bool[key][1]
            for key in self.targets_real.keys():
                self.sps_values[key] = self.targets_real[key][1]
            if write:
                self.write()

    def write_db(self):
        self.plc.db_write(self.db_set, 0, self.db)

    def read_db(self):
        self.db = self.plc.db_read(self.db_set, 0, 18)

    def write(self):
        if self.db is None:
            #print("NO DB AVAILABLE.")
            return

        for key in self.targets_bool.keys():
            snap7.util.set_bool(self.db,self.targets_bool[key][0][0], self.targets_bool[key][0][1], self.sps_values[key])

        for key in self.targets_real.keys():
            snap7.util.set_real(self.db, self.targets_real[key][0], self.sps_values[key])

        self.write_db()

    def read(self):
        if self.db is None:
            #print("NO DB AVAILABLE.")
            return
        self.sps_values = {}
        for key in self.targets_bool.keys():
            self.sps_values[key] = snap7.util.get_bool(self.db,self.targets_bool[key][0][0], self.targets_bool[key][0][1])

        for key in self.targets_real.keys():
            self.sps_values[key] = snap7.util.get_real(self.db,self.targets_real[key][0])