import unittest
import numpy as np
from shutil import rmtree

from tifuun_sensitivity import cascade
from tifuun_sensitivity.data import DESHIMA2

FOLDER = "_test"

class Test_Cascade(unittest.TestCase):
    def test_yaml_write_read(self):
        cascade.save_cascade(DESHIMA2.cascade_list,
                             FOLDER)        

        test_list = cascade.read_from_folder(FOLDER)

        for stage_ori, stage_load in zip(DESHIMA2.cascade_list, test_list):
            for key_ori, item_ori in stage_ori.items():
                for key_load, item_load in stage_load.items():
                    if key_ori == key_load:
                        if isinstance(item_ori, tuple):
                            self.assertTrue(np.allclose(item_ori[0],
                                                        item_load[0]))

                            self.assertTrue(np.allclose(item_ori[1],
                                                        item_load[1]))
                        else:
                            self.assertAlmostEqual(item_ori, 
                                                   item_load)

        rmtree(FOLDER)        

if __name__ == "__main__":
    import nose2
    nose2.main()
