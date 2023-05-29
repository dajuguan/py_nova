from py_ecc.fields.field_elements import FQ, Union
from dataclasses import dataclass
from typing import List,Type, TypeVar
import unittest



T_VecFQ = TypeVar('T_VecFQ', bound="VecFQ")

@dataclass
class VecFQ(object):
    """
    A class for field elements in FQ. Wrap a number in this class,
    and it becomes a field element.
    """
    vals: List[FQ]  # type: int
    def __add__(self: T_VecFQ, other: T_VecFQ) -> T_VecFQ:
        r = []
        for i in range(len(self.vals)):
            r.append(int(self.vals[i]) + int(other.vals[i]))  #must be int r to satisfy the ecc additive homomorphism
        return VecFQ(r)

    def __sub__(self: T_VecFQ, other: T_VecFQ) -> T_VecFQ:
        r = []
        for i in range(len(self.vals)):
            r.append(self.vals[i] - other.vals[i])
        return VecFQ(r)
    def __mul__(self: T_VecFQ, e: Union[int,FQ, T_VecFQ]) -> T_VecFQ:
        r = []
        if isinstance(e, int) or isinstance(e, FQ):
            for i in range(len(self.vals)):
                r.append( int(self.vals[i]) * int(e))   #must be int r to satisfy the ecc additive homomorphism
        elif isinstance(e, VecFQ):
            for i in range(len(e)):
                r.append( self.vals[i]*e.vals[i])
        else:
            assert("e must be int|FQ|VecFQ")
        return VecFQ(r)

    def __getitem__(self,key) -> FQ:
        return self.vals[key]
    def __len__(self) -> int:
        return len(self.vals)


def matrix_vector_product(M:List[List[FQ]], z: List[FQ]) -> T_VecFQ:
    FQ_Type = z[0].__class__
    r = [FQ_Type(0)]*len(M)
    for i in range(len(M)):
        for j in range(len(M[i])):
            if M[i][j] == 0:  #simulate sparse opearation
                continue
            r[i] += M[i][j] * z[j]
    return VecFQ(r)

# def hadamard_product(a: List[FQ], b: List[FQ]) -> T_VecFQ:
#     r = []
#     for i in range(len(a)):
#         r.append( a[i]*b[i])
#     return VecFQ(r)


def to_FQ_matrix(M: List[List[int]], FQ_Type:FQ)-> List[List[FQ]]:
    for i in range(len(M)):
        for j in range(len(M[i])):
            M[i][j] = FQ_Type( M[i][j])
    return M

def to_FQ_vec(z: List[int], FQ_Type:FQ) -> T_VecFQ:
    for i in range(len(z)):
        z[i] = FQ_Type(z[i])
    return VecFQ(z)

# def vector_elem_product(z: List[FQ],e:FQ)-> T_VecFQ:
#     assert (type(e) == type(z[0]))
#     r = []
#     for i in range(len(z)):
#         r.append( z[i] * e)
#     return VecFQ(r)





from pallas import Fp
class Test(unittest.TestCase):
    def test_matrix_vector_product(self):
        A = to_FQ_matrix([
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [5, 0, 0, 0, 0, 1]
        ],Fp)
        z = to_FQ_vec(
            [1, 3, 35, 9, 27, 30],
            Fp
        )
        Az = matrix_vector_product(A, z)
        assert (Az == to_FQ_vec([3,9,30,35], Fp))
    def test_hadamard_product(self):
        a = to_FQ_vec([1, 2, 3, 4, 5, 6], Fp)
        b = to_FQ_vec([7, 8, 9, 10, 11, 12], Fp)
        assert (a * b == to_FQ_vec([7, 16, 27, 40, 55, 72], Fp))
    def test_ABC_hadamard(self):
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
        z = to_FQ_vec([1, 3, 35, 9, 27, 30], Fp)
        Az = matrix_vector_product(A,z)
        Bz = matrix_vector_product(B,z)
        Cz = matrix_vector_product(C,z)
        # print("Cz------------->", Cz, Cz[0]- Fp(10))
        assert ( Az * Bz == Cz)
    def test_vector_elem_product(self):
        z = to_FQ_vec([1, 3, 35, 9, 27, 30], Fp)
        u = Fp(3)
        r = z*u
        assert (r == to_FQ_vec([3, 9, 105, 27, 81, 90], Fp))
    def test_VecFQ(self):
        a = to_FQ_vec([1, 3, 35, 9, 27, 30], Fp)
        b = to_FQ_vec([1, 1, 1, 1, 1, 1], Fp)
        c = VecFQ(vals=a) + VecFQ(vals=b)
        assert (c == VecFQ([2, 4, 36, 10, 28, 31]))
        d = c - VecFQ(vals=b)
        assert (d == VecFQ([1, 3, 35, 9, 27, 30]))


if __name__ == '__main__':
    unittest.main()
