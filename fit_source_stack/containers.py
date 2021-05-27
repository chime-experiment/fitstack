from caput import memh5

from draco.core.containers import *

class MockArray(memh5.BasicCont):

    convert_attribute_strings = True
    convert_dataset_strings = True

    @property
    def freq(self):
        return self.index_map['freq']['centre'][:]

    @property
    def pol(self):
        return self.index_map['pol'][:]

    @property
    def stack(self):
        return self['stack']

    @property
    def weight(self):
        return self['weight']