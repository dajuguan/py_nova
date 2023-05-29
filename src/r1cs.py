from dataclasses import dataclass
from typing import List,Tuple
import unittest
from py_ecc.fields.field_elements import FQ
from pallas import Fp, PallaCurve
from pederson import CommitmentKey, commit,Commitment, Proof
import pederson
from utils import matrix_vector_product, VecFQ
import copy
from transcript import Transcript

@dataclass
class R1CSInstance:
    comm_W: any
    X: List[FQ]

@dataclass
class R1CS:
    A: List[List[FQ]]
    B: List[List[FQ]]
    C: List[List[FQ]]
    ck = None  #commitment_key

    # num_cons: int
    # num_vars: int
    # num_io: int
    # digest: str

    # def __init__(self,num_cons,num_vars, num_io, A, B, C):
    #     self.num_cons = num_cons
    #     self.num_vars = num_vars
    #     self.num_io = num_io
    #     self.A = A
    #     self.B = B
    #     self.C = C
    #     # todo compute
    #     self.digest = "ABC"  
    def commitment_key(self) -> CommitmentKey:
        total_nz = len(self.A) + len(self.B) + len(self.C)
        g = PallaCurve.generator()
        h = g * (Fp.rand())
        # h = PallaCurve((Fp(14461209976817224340751362617869713747405720547434143279679227140820555473247), Fp(71665485320935192427476043561349813289436657151049165109128120723998422331)))
        # ck = CommitmentKey(PallaCurve.from_label(b"ck",total_nz),g,h)
        generators = [g*Fp.rand() for _ in range(total_nz)]
        ck = CommitmentKey(generators, g, h)
        self.ck = ck
        return ck

@dataclass
class RelaxedR1CSInstance:
    commitmentE: Commitment
    u: FQ
    commitmentW: Commitment
    x: FQ

@dataclass
class FoldedWitness:
    E: VecFQ
    rE: FQ
    W: VecFQ
    rW: FQ

    def commit(self, ck: CommitmentKey) -> RelaxedR1CSInstance:
        commitmentE = commit(ck,self.E, self.rE)
        commitmentW = commit(ck,self.W, self.rW)
        u = self.rE.__class__(1)
        x = self.W[0]    #todo
        return RelaxedR1CSInstance(commitmentE,u,commitmentW,x)

# construction 2
class NIFS:
    def compute_T(r1cs: R1CS, u1: FQ, u2: FQ, z1: List[FQ], z2: List[FQ]) -> VecFQ:
        A, B, C = r1cs.A, r1cs.B, r1cs.C
        Az1 = matrix_vector_product(A, z1)
        Bz1 = matrix_vector_product(B, z1)
        Cz1 = matrix_vector_product(C, z1)
        Az2 = matrix_vector_product(A, z2)
        Bz2 = matrix_vector_product(B, z2)
        Cz2 = matrix_vector_product(C, z2)
        Az1_Bz2 = Az1 * Bz2
        Az2_Bz1 = Az2 * Bz1
        u1_Cz2 = Cz2 * u1
        u2_Cz1 = Cz1 * u2
        T = Az1_Bz2 + Az2_Bz1 - u1_Cz2 - u2_Cz1
        return T
    def fold_witness(r: FQ, wit1: FoldedWitness, wit2: FoldedWitness, T: VecFQ, rT: FQ) -> FoldedWitness:
        r_square = r * r
        E = wit1.E + T*r + wit2.E*r_square
        # print("wit1.E---------------->", len(wit1.E),len(E))
        rE = wit1.rE + rT*r + wit2.rE*r_square
        W = wit1.W + wit2.W * r
        rW = wit1.rW + wit2.rW * r
        return FoldedWitness(E,rE,W,rW)
    def fold_instance(
            r: FQ, 
            ins1: RelaxedR1CSInstance, 
            ins2: RelaxedR1CSInstance,
            cmT: Commitment
    ) -> RelaxedR1CSInstance:
        r_square = r * r
        cmE = ins1.commitmentE + cmT*r + ins2.commitmentE*r_square
        u = ins1.u + ins2.u*r
        cmW = ins1.commitmentW + ins2.commitmentW*r
        x = ins1.x + ins2.x*r
        return RelaxedR1CSInstance(cmE, u, cmW, x)
    def is_circuit_satisfied(r1cs: R1CS, ins: RelaxedR1CSInstance, wit:FoldedWitness) -> bool:
        # verify if Az * Bz = u*Cz + E
        A,B,C = r1cs.A, r1cs.B, r1cs.C
        z, u = wit.W, ins.u
        Az = matrix_vector_product(A,z)
        Bz = matrix_vector_product(B,z)
        Cz = matrix_vector_product(C,z)
        # print("len(Cz), len(with.E)------------>",len(Az*Bz) ,len(Cz), len(wit.E))
        return Az * Bz == Cz*u + wit.E

    def open_commitments(
            ts:Transcript,
            ck:CommitmentKey,
            fw:FoldedWitness,
            ins:RelaxedR1CSInstance,
            T: VecFQ,
            rT: FQ,
            cmT: Commitment
            ) -> Tuple[Proof,Proof,Proof]:
        cmE_proof = pederson.prove(ck, ts, ins.commitmentE, fw.E, fw.rE)
        cmW_proof = pederson.prove(ck, ts, ins.commitmentW, fw.W, fw.rW)
        cmT_proof = pederson.prove(ck, ts, cmT, T, rT)
        return (cmE_proof, cmW_proof, cmT_proof)
    
    def verify_commitments(
        ts: Transcript,
        ck: CommitmentKey,
        ins: RelaxedR1CSInstance,
        cmT: Commitment,
        cmE_proof: Proof,
        cmW_proof: Proof,
        cmT_proof: Proof
    ) -> bool:
        a = pederson.verify(ck,ts,ins.commitmentE,cmE_proof) 
        b = pederson.verify(ck,ts,ins.commitmentW,cmW_proof) 
        c = pederson.verify(ck,ts,cmT,cmT_proof) 
        return a and b and c


class TestR1CS(unittest.TestCase):
    def setUp(self): #every function will invoke this, thus r1cs will be the same with setUp
        from utils import to_FQ_matrix, to_FQ_vec
        A = to_FQ_matrix([
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [5, 0, 0, 0, 0, 1]
        ],Fp)
        B = to_FQ_matrix([
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0]
        ],Fp)
        C = to_FQ_matrix([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0]
        ],Fp)
        # generate n witness
        witnesses: List[VecFQ] = []
        for i in range(3):
            input = 3 + i
            w_i = to_FQ_vec([
                1,
                input,
                input * input * input + input + 5, # x^3 + x + 5
                input * input,                     # x^2
                input * input * input,             # x^2 * x
                input * input * input + input,     # x^3 + x
            ], Fp)
            witnesses.append(w_i)
        r1cs = R1CS(A,B,C)
        self.witnesses = witnesses
        self.r1cs = r1cs
        assert NIFS.is_circuit_satisfied(
            r1cs,
            RelaxedR1CSInstance(None,Fp(1),None,None), 
            FoldedWitness(
                to_FQ_vec([0]*len(A), Fp),
                None,
                witnesses[0],
                None)
            )
    def test_one_fold(self):
        r1cs, w1, w2 = self.r1cs, self.witnesses[0], self.witnesses[1]
        A,B,C = r1cs.A, r1cs.B, r1cs.C
        ck = r1cs.commitment_key()
        # initialize folded witness
        fw1 = FoldedWitness(VecFQ([Fp(0)]*len(A)), Fp(1), copy.deepcopy(w1), Fp(1))
        fw2 = FoldedWitness(VecFQ([Fp(0)]*len(A)), Fp(1), copy.deepcopy(w2), Fp(1))
        # get committed instance
        rins1 = fw1.commit(ck)
        rins2 = fw2.commit(ck)
        # prepare for fold
        T = NIFS.compute_T(r1cs,rins1.u,rins2.u,w1,w2)
        rT = 10 # change to rand, this would come from the transcript
        cmT = commit(ck, T, rT)
        # fold
        r = 2 # change to rand

        foldedIns = NIFS.fold_instance(r,rins1,rins2,cmT)
        foldedWit = NIFS.fold_witness(r,fw1,fw2,T,rT)

        ## check that the folded witness satisfies the relaxed r1cs
        assert NIFS.is_circuit_satisfied(r1cs,foldedIns, foldedWit)

        # ## check that foldedIns commitment is equal to use the folded rE, rW to commit
        foldedExpected = foldedWit.commit(ck)
        assert (foldedExpected.commitmentW == foldedIns.commitmentW )
        assert (foldedExpected.commitmentE == foldedIns.commitmentE)

        from poseidon import PoseidonConfig, PoseidonSponge

        sponge_config = PoseidonConfig.test_config()
        sponge = PoseidonSponge(sponge_config, Fp)

        # init Prover & Verifier's transcript
        transcript_p = Transcript(sponge)
        transcript_v = copy.deepcopy(transcript_p)

        # check openings of rins.cmE, rins.cmW and cmT
        (cmE_proof, cmW_proof, cmT_proof) = NIFS.open_commitments(
            transcript_p,
            ck,
            foldedWit,
            foldedIns,
            T,
            rT,
            cmT
        )

        assert NIFS.verify_commitments(
            transcript_v,
            ck,
            foldedIns,
            cmT,
            cmE_proof,
            cmW_proof,
            cmT_proof
        )
        print("Basic R1CS folding Test is OK!")

if __name__ == '__main__':
    unittest.main()
