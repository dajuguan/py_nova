from py_ecc.fields.field_elements import FQ as Field
import unittest
from Cryptodome.Hash import SHAKE256,BLAKE2b
import numpy as np
from hash_to_curve import hash_to_vesta_jacobian, hash_to_pallas_jacobian
from typing import List

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

from py_ecc.typing import (
    Field,
    GeneralPoint,
    Point2D,
)


p = Fp.field_modulus
curve_order = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001


# Curve is y**2 = x**3 + 5
b = Fp(5)

# Generator for curve over FQ
G1 = (Fp(-1), Fp(2))

# Point at infinity over FQ
Z1 = None

IsoEp_A = 10949663248450308183708987909873589833737836120165333298109615750520499732811
IsoEq_A = 17413348858408915339762682399132325137863850198379221683097628341577494210225
IsoEp_B = 1265
IsoEq_B = 1265
IsoEpZ = p -13
F_p = Fp(p)



# Check if a point is the point at infinity
def is_inf(pt: GeneralPoint[Field]) -> bool:
    return pt is None


# Check that a point is on the curve defined by y**2 == x**3 + b
def _is_on_curve(pt: Point2D[Field], b: Field) -> bool:
    if is_inf(pt):
        return True
    x, y = pt
    return y**2 - x**3 == b

def is_on_curve(pt: Point2D[Field]) -> bool:
    return _is_on_curve(pt, b)

# Elliptic curve doubling
def double(pt: Point2D[Field]) -> Point2D[Field]:
    if is_inf(pt):
        return pt
    x, y = pt
    m = 3 * x**2 / (2 * y)
    newx = m**2 - 2 * x
    newy = -m * newx + m * x - y
    return (newx, newy)


# Elliptic curve addition
def add(p1: Point2D[Field],
        p2: Point2D[Field]) -> Point2D[Field]:
    if p1 is None or p2 is None:
        return p1 if p2 is None else p2
    x1, y1 = p1
    x2, y2 = p2
    if x2 == x1 and y2 == y1:
        return double(p1)
    elif x2 == x1:
        return None
    else:
        m = (y2 - y1) / (x2 - x1)
    newx = m**2 - x1 - x2
    newy = -m * newx + m * x1 - y1
    assert newy == (-m * newx + m * x2 - y2)
    return (newx, newy)


# Elliptic curve point multiplication
def multiply(pt: Point2D[Field], n: int) -> Point2D[Field]:
    if n == 0:
        return None
    elif n == 1:
        return pt
    elif not n % 2:
        return multiply(double(pt), n // 2)
    else:
        return add(multiply(double(pt), int(n // 2)), pt)


def eq(p1: GeneralPoint[Field], p2: GeneralPoint[Field]) -> bool:
    return p1 == p2


# Convert P => -P
def neg(pt: Point2D[Field]) -> Point2D[Field]:
    if pt is None:
        return None
    x, y = pt
    return (x, -y)


### setup related
def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()



def from_label(label, n):
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
        (p_affine, p_jacob, c) = hash_to_pallas_jacobian(msg, b"from_uniform_bytes")
        (x,y,_) = p_affine #(x,y,1)
        preprocessedGroupElement.append((Fp(int(x)),Fp(int(y))))
    return preprocessedGroupElement

def isop_map_jacobian(P):
    (x, y, z, z2, z3) = (P.x, P.y, P.z, P.z2, P.z3)
    z4 = z2**2
    z6 = z3**2

    Nx = ((( 6432893846517566412420610278260439325191790329320346825767705947633326140075    *x +
            23989696149150192365340222745168215001509815558210986772351135915822265203574*z2)*x +
            10492611921771203378452795982353351666191589197598957448093274638589204800759*z4)*x +
            12865787693035132824841220556520878650383580658640693651535411895266652280192*z6)

    Dx =  ((                                                                              z2 *x +
            13271109177048389296812780941310096270046944650307955939477485891950613419807*z4)*x +
            22768321103861051515190775253992702316905399997697804654926324362758820947460*z6)


    Ny = (((11793638718615538422771118843477472096184948937087302513907460903994431256804    *x +
            11994848074575096182670111372584107500754907779105493386175567957911132601787*z2)*x +
            28823569610051396102362669851238297121581474897215657071023781420043761726004*z4)*x +
             1072148974419594402070101713043406554198631721553391137627950991272221023311*z6) * y

    Dy = (((                                                                                  x +
             5432652610908059517272798285879155923388888734491153551238890455750936314542*z2)*x +
            10408918692925056833786833257634153023990087029210292532869619559576527581706*z4)*x +
            28948022309329048855892746252171976963363056481941560715954676764349967629797*z6) * z3


    zo = Dx*Dy
    xo = Nx*Dy * zo
    yo = Ny*Dx*zo

    return (xo, yo, zo)

class PallasTest(unittest.TestCase):
    def test_math(self):
        g2 = add(G1,G1)
        _g2 = multiply(G1,2)
        assert(g2 == _g2)
    def test_shake(self):  # compatible with Nova 
        shake = SHAKE256.new()
        shake.update(b"test_from_label")
        hash = shake.read(8)
        res = np.frombuffer(hash, dtype=np.uint8)
        assert (res == [96, 43, 193, 79, 100, 95, 20, 71]).all()
    def test_from_lable(self):
        commitKeys = from_label(b"ck",3)
        print("commitKeys---------->", commitKeys)
    def test_is_on_curve(self):
        commitKeys = from_label(b"ck",1)
        (x,y) = commitKeys[0]
        x, y = int(x), int(y)
        assert is_on_curve((Fp(x),Fp(y)))



if __name__ == '__main__':
    unittest.main()
