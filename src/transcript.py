# simulate the interaction between prover and verifier
from dataclasses import dataclass
from utils import VecFQ
from pallas import CurveGroup
from poseidon import PoseidonSponge
from py_ecc.typing import FQ
import unittest

@dataclass
class Transcript:
    sponge: PoseidonSponge

    def get_challenge(self) -> FQ:
        c = self.sponge.squeeze_field_elements(1)
        self.sponge.absorb(c)
        return c[0]
    def get_challenge_vec(self, n: int) -> VecFQ:
        c = self.sponge.squeeze_field_elements(n)
        self.sponge.absorb(c)
        return VecFQ(c)
    def add_point(self, p: CurveGroup):
        (x,y) = p.pt
        self.sponge.absorb([x,y])

class Test(unittest.TestCase):
    def test_transcript(self):
        from src.poseidon import PoseidonConfig, PoseidonSponge
        from pallas import Fp, PallaCurve
        sponge = PoseidonSponge(PoseidonConfig.test_config(), Fp)

        ts = Transcript(sponge)
        ts.add_point(PallaCurve((Fp(-1), Fp(2))))
        c = ts.get_challenge()
        assert c== Fp(11277200730346613550085271047300679821640671579635075722393542866312425805972)
        c = ts.get_challenge_vec(2)
        assert c== [Fp(10580274181532375282948200310297166284925572589287530674403852134840241606569), Fp(3402551684005499196345148666259244740440101670427500285000912632282126274286)]

if __name__ == '__main__':
    unittest.main()