import unittest
from typing import List
from py_ecc.typing import Field,Point2D
from curve import ec_lincomb
import pallas

def commit(commitmentKey:List[Point2D], val:List[Field],curve_type="pallas"):
    if curve_type == "pallas":
        commitment = ec_lincomb([(x,c) for x,c in zip(commitmentKey[:len(val)], val)],pallas)
    else:
        commitment = ec_lincomb([(x,c) for x,c in zip(commitmentKey[:len(val)], val)],pallas)
    return commitment

class Test(unittest.TestCase):
    def test_commit(self):
        from pallas import from_label,Fp,is_on_curve
        commitKeys = from_label(b"ck",3)
        commitment = commit(commitKeys, [Fp(2),Fp(3),Fp(4)],"pallas")
        print("commitment------------>", commitment)
        assert is_on_curve(commitment)

if __name__ == '__main__':
    unittest.main()