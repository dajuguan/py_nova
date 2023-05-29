from py_ecc.fields.field_elements import FQ as Field
from dataclasses import dataclass
import unittest
from Cryptodome.Hash import SHAKE256,BLAKE2b
# from hash_to_curve import hash_to_vesta_jacobian, hash_to_pallas_jacobian
from typing import List, TypeVar, Callable, Union
from py_ecc.typing import Point2D
import copy
import random


class Fp(Field):
    field_modulus = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
    @classmethod
    def root_of_unity(cls, group_order: int):
        return Fp(5) ** ((cls.field_modulus - 1) // group_order)

    # Gets the full list of roots of unity of a given group order
    @classmethod
    def roots_of_unity(cls, group_order: int):
        o = [Fp(1), cls.root_of_unity(group_order)]
        while len(o) < group_order:
            o.append(o[-1] * o[1])
        return o
    
    @classmethod
    def rand(cls) -> Field:
        r = random.randrange(0,cls.field_modulus)
        return Fp(r)
    

class Fq(Field):
    field_modulus = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001


T_CurveGroup = TypeVar("T_CurveGroup", bound="CurveGroup")
@dataclass
class CurveGroup:
    pt: Point2D
    Field_Type: Field
    # field_modulus: int
    curve_order: int
    b: Field
    # hash_to_curve: Callable
    # Point at infinity over FQ
    Z1 = None

    def __init__(self, pt):
        self.pt = pt

    @classmethod
    def generator(cls) -> T_CurveGroup:
        p = (cls.Field_Type(-1),cls.Field_Type(2))
        return cls(p)

    def is_inf(self) -> bool:
        return self.pt is None
    
    # Check that a point is on the curve defined by y**2 == x**3 + b
    def is_on_curve(self) -> bool:
        pt = self.pt
        if self.is_inf():
            return True
        x, y = pt
        return y**2 - x**3 == self.b

    # Elliptic curve doubling
    def double(self) -> T_CurveGroup:
        pt = self.pt
        if self.is_inf():
            return self.__class__(pt)
        x, y = pt
        m = 3 * x**2 / (2 * y)
        newx = m**2 - 2 * x
        newy = -m * newx + m * x - y
        pt = (newx, newy)
        return self.__class__(pt)

    # Elliptic curve addition
    def __add__(self, other: T_CurveGroup) -> T_CurveGroup:
        p1, p2 = self.pt, other.pt
        if p1 is None or p2 is None:
            return copy.deepcopy(self) if p2 is None else copy.deepcopy(other)
        x1, y1 = p1
        x2, y2 = p2
        if x2 == x1 and y2 == y1:
            return copy.deepcopy(self).double()
        elif x2 == x1:
            return self.__class__(None)
        else:
            m = (y2 - y1) / (x2 - x1)
        newx = m**2 - x1 - x2
        newy = -m * newx + m * x1 - y1
        assert newy == (-m * newx + m * x2 - y2)
        pt = (newx, newy)
        return self.__class__(pt)

    # Elliptic curve point multiplication
    def __multiply(self, n: int) -> T_CurveGroup:
        if n == 0:
            return self.__class__(None)
        elif n == 1:
            return copy.deepcopy(self)
        elif not n % 2:
            return self.double().__multiply(n // 2)
        else:
            return self.double().__multiply(int(n // 2)) + self
    def __mul__(self, n:Union[int,Field]) -> T_CurveGroup:
        if isinstance(n, Field):
            return self.__multiply(int(n))
        else:
            return self.__multiply(n % self.curve_order)

    def __eq__(self, other: T_CurveGroup) -> bool:
        return self.pt == other.pt

    # Convert P => -P
    def __neg__(self) -> T_CurveGroup:
        pt = self.pt
        if pt is None:
            return None
        x, y = pt
        return self.__class__((x, -y))
    
    @classmethod
    def from_label(cls,label, n) -> List[T_CurveGroup]:
        Field_Type = cls.Field_Type
        # expand_message_xmd
        shake = SHAKE256.new()
        shake.update(label)
        uniform_bytes_vec = []
        for _ in range(n):
            hash =  shake.read(32)
            # hash_bytes = np.frombuffer(hash, dtype=np.uint8)
            uniform_bytes_vec.append(hash)
        preprocessedGroupElement = []
        for msg in uniform_bytes_vec:
            (p_affine, p_jacob, c) = cls.hash_to_curve(msg, b"from_uniform_bytes")
            (x,y,_) = p_affine #(x,y,1)
            p = cls((Field_Type(int(x)),Field_Type(int(y))))
            preprocessedGroupElement.append(p)
        return preprocessedGroupElement
    
    @classmethod
    def ec_lincomb(cls, pts: List[T_CurveGroup], vals:  List[int] )-> T_CurveGroup:
        p = lincomb(pts, vals, cls.__add__, pts[0].__class__(cls.Z1))
        return p
        # Equivalent to:
            # o = b.Z1
            # for pt, coeff in pairs:
            #     o = b.add(o, ec_mul(pt, coeff))
            # return o


################################################################
# multicombs
################################################################

import random, sys, math
def multisubset(numbers, subsets, adder=lambda x, y: x + y, zero=0):
    # Split up the numbers into partitions
    partition_size = 1 + int(math.log(len(subsets) + 1))
    # Align number count to partition size (for simplicity)
    numbers = numbers[::]
    while len(numbers) % partition_size != 0:
        numbers.append(zero)
    # Compute power set for each partition (eg. a, b, c -> {0, a, b, a+b, c, a+c, b+c, a+b+c})
    power_sets = []
    for i in range(0, len(numbers), partition_size):
        new_power_set = [zero]
        for dimension, value in enumerate(numbers[i : i + partition_size]):
            r = [adder(n, value) for n in new_power_set]
            # new_power_set.append([adder(n, value) for n in new_power_set][0])
            new_power_set += [adder(n, value) for n in new_power_set]
        power_sets.append(new_power_set)
    # Compute subset sums, using elements from power set for each range of values
    # ie. with a single power set lookup you can get the sum of _all_ elements in
    # the range partition_size*k...partition_size*(k+1) that are in that subset
    subset_sums = []
    for subset in subsets:
        o = zero
        for i in range(len(power_sets)):
            index_in_power_set = 0
            for j in range(partition_size):
                if i * partition_size + j in subset:
                    index_in_power_set += 2**j
            o = adder(o, power_sets[i][index_in_power_set])
        subset_sums.append(o)
    return subset_sums


# Reduces a linear combination `numbers[0] * factors[0] + numbers[1] * factors[1] + ...`
# into a multi-subset problem, and computes the result efficiently
def lincomb(numbers, factors, adder=lambda x, y: x + y, zero=0):
    # Maximum bit length of a number; how many subsets we need to make
    maxbitlen = max(len(bin(f)) - 2 for f in factors)
    # Compute the subsets: the ith subset contains the numbers whose corresponding factor
    # has a 1 at the ith bit
    subsets = [
        {i for i in range(len(numbers)) if factors[i] & (1 << j)}
        for j in range(maxbitlen + 1)
    ]
    subset_sums = multisubset(numbers, subsets, adder=adder, zero=zero)
    # For example, suppose a value V has factor 6 (011 in increasing-order binary). Subset 0
    # will not have V, subset 1 will, and subset 2 will. So if we multiply the output of adding
    # subset 0 with twice the output of adding subset 1, with four times the output of adding
    # subset 2, then V will be represented 0 + 2 + 4 = 6 times. This reasoning applies for every
    # value. So `subset_0_sum + 2 * subset_1_sum + 4 * subset_2_sum` gives us the result we want.
    # Here, we compute this as `((subset_2_sum * 2) + subset_1_sum) * 2 + subset_0_sum` for
    # efficiency: an extra `maxbitlen * 2` group operations.
    o = zero
    for i in range(len(subsets) - 1, -1, -1):
        o = adder(adder(o, o), subset_sums[i])
    return o


class PallaCurve(CurveGroup):
    pt: Point2D
    Field_Type = Fp
    # Curve is y**2 = x**3 + 5
    b = Fp(5)
    # hash_to_curve = hash_to_pallas_jacobian
    curve_order = Fq.field_modulus

# @dataclass
class VestaCurve(CurveGroup):
    pt: Point2D
    Field_Type = Fq
    b = Fq(5)
    # hash_to_curve = hash_to_vesta_jacobian
    curve_order = Fp.field_modulus


### setup related
def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

class Test(unittest.TestCase):
    def test_pallas_add_mul_double_neg(self):
        point = (Fp(-1), Fp(2))
        g1 = PallaCurve(point)
        g2 = g1 + g1
        _g2 = g1 * 2
        _g3 = g1.double()
        # print("g1 pallas---------->", g2)
        assert (g2 == _g2 == _g3)
        negPoint = PallaCurve((Fp(-1),Fp(-2)))
        assert (-g1 == negPoint)
        assert g2.is_on_curve()
    def test_vesta_add_mul_double_neg(self):
        point = (Fq(-1), Fq(2))
        g1 = VestaCurve(point)
        g2 = g1 + g1
        _g2 = g1 * 2
        _g3 = g1.double()
        # print("g2 vesta---------->", g2)
        assert (g2 == _g2 == _g3)
        negPoint = VestaCurve((Fq(-1),Fq(-2)))
        assert (-g1 == negPoint)
        assert g2.is_on_curve()
    # def test_from_label(self):
    #     pallas_commitKeys = PallaCurve.from_label(b"ck",3)
    #     # print("commitKeys---------->", pallas_commitKeys)
    #     p = PallaCurve(pallas_commitKeys[0])
    #     assert p.is_on_curve()
    #     vesta_commitKeys = PallaCurve.from_label(b"ck",3)
    #     q = PallaCurve(vesta_commitKeys[0])
    #     assert q.is_on_curve()
    def test_ec_lincomb(self):
        point = (Fq(-1), Fq(2))   #the point must be on curve first
        g1 = VestaCurve(point)
        pts = [g1,g1*2,g1*3]
        vals = [1,2,3]
        r = VestaCurve.ec_lincomb(pts,vals)
        assert r == g1*14
        assert r.is_on_curve()


if __name__ == '__main__':
    unittest.main()
