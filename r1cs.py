from dataclasses import dataclass
from typing import List,Tuple
import unittest
from py_ecc.fields.field_elements import FQ
from curve import ec_lincomb
from pallas import Fp,from_label

@dataclass
class R1CSInstance:
    comm_W: any
    X: List[FQ]

@dataclass
class CommitmentKey:
    ck: List[FQ]

@dataclass
class R1CSWitness:
    W: List[FQ]
    Curve: any
    def commit(self, commimentKey:CommitmentKey):
        commitment = ec_lincomb([(x,c) for x,c in zip(self.W, commimentKey.ck)], self.Curve)
        return commitment
    


@dataclass
class R1CS:
    num_cons: int
    num_vars: int
    num_io: int
    A: Tuple[int,int,FQ]
    B: Tuple[int,int,FQ]
    C: Tuple[int,int,FQ]
    digest: str

    def __init__(self,num_cons,num_vars, num_io, A, B, C):
        self.num_cons = num_cons
        self.num_vars = num_vars
        self.num_io = num_io
        self.A = A
        self.B = B
        self.C = C
        # todo compute
        self.digest = "ABC"  
    def commitment_key(self):
        total_nz = len(self.A) + len(self.B) + len(self.C)
        return from_label(b"ck", max(self.num_cons, self.num_vars, self.num_io,total_nz) )

        
    def multiply_vec(self, z:FQ):
        def sparce_matrix_vec_product(M:Tuple[int,int,FQ]):
            Mz = FQ.zero()
            for (row, col,val) in M:
                Mz += val * z[col]
        Az = sparce_matrix_vec_product(self.A, z)
        Bz = sparce_matrix_vec_product(self.B, z)
        Cz = sparce_matrix_vec_product(self.C, z)
        return (Az, Bz, Cz)
    def is_sat(self, ck, U:R1CSInstance, W:R1CSWitness):
        # verify if Az * Bz = u*Cz
        pass




        


class TestR1CS(unittest.TestCase):
    def test_init(self):
        r1cs = R1CS(1,2,3,(1,1,Fp(1)),(2,2,Fp(1)),(3,3,Fp(1)))
        assert(r1cs.num_cons == 1)

if __name__ == '__main__':
    unittest.main()
