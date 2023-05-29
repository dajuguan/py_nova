import unittest
from typing import List,Type, TypeVar,Union
from py_ecc.typing import Field,Point2D,FQ
from dataclasses import dataclass
from pallas import CurveGroup
from transcript import Transcript
from utils import VecFQ
import numpy as np

@dataclass 
class CommitmentKey:
    generators: List[CurveGroup]
    g: CurveGroup
    h: CurveGroup
T_Commitment =  TypeVar('T_Commitment', bound="Commitment")

@dataclass
class Proof:
    R_cm: CurveGroup
    u_: List[FQ]
    ru_: FQ

@dataclass
class Commitment:
    c: CurveGroup
    def __add__(self: T_Commitment, other: T_Commitment) -> T_Commitment:
        return Commitment(self.c + other.c)
    def __mul__(self: T_Commitment, other: FQ) -> T_Commitment:
        return Commitment(self.c * other)
    def __eq__(self, other: T_Commitment) -> bool:
        return self.c == other.c
        


def commit(ck:CommitmentKey, vals:List[FQ], r: Union[int,FQ]) -> Commitment:
    CurveType:CurveGroup = ck.g.__class__    
    _vals = [int(x) % CurveType.curve_order for x in vals]
    p = ck.h*int(r) +  CurveType.ec_lincomb(ck.generators[:len(vals)], _vals)
    return Commitment(p)

def prove(ck:CommitmentKey,ts:Transcript, cm: Commitment, v: VecFQ, r: Field) -> Proof:
    r1 = ts.get_challenge()
    d = ts.get_challenge_vec(len(v))
    R_cm = commit(ck,d,int(r1))
    
    ts.add_point(cm.c)
    ts.add_point(R_cm.c) 
    e = ts.get_challenge()

    uu = (v*e + d).vals
    v = np.array([int(x) for x in v.vals])
    ru_ = int(e)*int(r) + int(r1)

    return Proof(R_cm,uu,ru_)

def verify(ck:CommitmentKey,ts:Transcript,cm: Commitment,pf: Proof) -> bool:
    # r1, d just to match Prover's transcript
    ts.get_challenge()
    d = ts.get_challenge_vec(len(pf.u_))

    ts.add_point(cm.c)
    ts.add_point(pf.R_cm.c)
    e = ts.get_challenge()

    lhs = pf.R_cm + cm*e
    lhs = pf.R_cm + cm*int(e)
    rhs = commit(ck, pf.u_, pf.ru_)

    return lhs == rhs



class Test(unittest.TestCase):
    def setUp(self):
        from pallas import PallaCurve, Fp, Fq
        g = PallaCurve.generator()
        h = g * (Fp.rand())
        h = PallaCurve((Fp(14461209976817224340751362617869713747405720547434143279679227140820555473247), Fp(71665485320935192427476043561349813289436657151049165109128120723998422331)))
        # ck = CommitmentKey(PallaCurve.from_label(b"ck",12), g, h)
        ck = CommitmentKey([g*Fp.rand() for _ in range(12)], g, h)
        self.ck = ck 
    def test_commit(self):
        from pallas import Fp

        ck = self.ck
        # r = Fp.rand()
        rT = 10
        # r = int(Fp.rand())
        r = 2
        commitment = commit(ck, [28948022309329048855892746252171976963363056481941560715954676764349967630336, 28948022309329048855892746252171976963363056481941560715954676764349967630330, Fp(0), Fp(0)],rT)
        # print("commitment------------>", commitment.c.pt)
        assert commitment.c.is_on_curve()
        # test additive homomorphism
        cm_2 = commit(ck, [28948022309329048855892746252171976963363056481941560715954676764349967630336*r, 28948022309329048855892746252171976963363056481941560715954676764349967630330*r, Fp(0)*r, Fp(0)*r],rT*r)
        cm_3 = commitment*r
        # print("cm_2", cm_2.c.pt)
        assert cm_2 == cm_3
    
    def test_prove_verify(self):
        from pallas import Fp
        from src.poseidon import PoseidonConfig, PoseidonSponge
        import copy

        ck = self.ck
        cm = commit(ck, [Fp(2), Fp(3)],2)

        sponge_config = PoseidonConfig.test_config()
        sponge = PoseidonSponge(sponge_config, Fp)
        ts = Transcript(sponge)
        _ts = copy.deepcopy(ts)
        pf = prove(ck,ts, cm, VecFQ([Fp(2),Fp(3)]), 2)
        assert verify(ck,_ts,cm,pf)

if __name__ == '__main__':
    unittest.main()