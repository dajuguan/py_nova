from dataclasses import dataclass
from typing import List, Union
from py_ecc.typing import FQ
from enum import Enum
from pallas import CurveGroup, Fp
from utils import VecFQ, matrix_vector_product, to_FQ_matrix
import unittest
import numpy as np
import math


@dataclass
class PoseidonConfig:
    # Number of rounds in a full-round operation.
    full_rounds: int
    # Number of rounds in a partial-round operation.
    partial_rounds: int
    # Exponent used in S-boxes.
    alpha: int
    # Additive Round keys. These are added before each MDS matrix application to make it an affine shift.
    # They are indexed by `ark[round_num][state_element_index]`
    ark: List[List[FQ]]
    # Maximally Distance Separating (MDS) Matrix.
    mds: List[List[FQ]]
    # The rate (in terms of number of field elements).
    # See [On the Indifferentiability of the Sponge Construction](https:#iacr.org/archive/eurocrypt2008/49650180/49650180.pdf)
    # for more details on the rate and capacity of a sponge.
    rate: int
    # The capacity (in terms of number of field elements).
    capacity: int

    @classmethod
    def test_config(cls):
        capacity = 100
        full_rounds = 8
        partial_rounds = 61
        p = Fp.field_modulus
        field_p = Fp
        alpha = 5
        rate = 20
        prime_bit_len = math.ceil(math.log2(p))
        ark = calc_round_constants(capacity + rate,full_rounds,partial_rounds,p,field_p,alpha,prime_bit_len)
        mds = mds_matrix_generator(Fp, capacity + rate)
        return PoseidonConfig(full_rounds, partial_rounds, alpha, ark, mds, rate, capacity)


class DuplexSpongeMode(Enum):
    Absorbing = (0)
    Squeezing = (0)

    def __init__(self, next_squeeze_index: int):
        self.next_squeeze_index = next_squeeze_index
        super().__init__()


class PoseidonSponge:
    config: PoseidonConfig
    state: List[FQ]
    mode: DuplexSpongeMode
    F: FQ
    def __init__(self, config: PoseidonConfig, fieldType: FQ):
        self.config = config
        self.F = fieldType
        self.state = [self.F.zero()]*(config.rate + config.capacity)
        self.mode = DuplexSpongeMode.Absorbing
    
    def absorb(self, input: List[FQ]) -> None:
        assert len(input) > 0
        match self.mode:
            case DuplexSpongeMode.Absorbing:
                absorb_index = self.mode.Absorbing.next_squeeze_index
                if absorb_index == self.config.rate:
                    self.permute()
                    absorb_index = 0
                self.absorb_internal(absorb_index, input[:])   #need to set absorb index back
            case DuplexSpongeMode.Squeezing:
                self.permute()
                self.absorb_internal(0, input[:])

    def permute(self):
        full_rounds_over_2 = int(self.config.full_rounds /2 )
        partial_rounds = self.config.partial_rounds
        full_rounds = self.config.full_rounds

        for i in range(full_rounds_over_2):
            self.apply_ark(i)
            self.apply_s_box(True)
            self.apply_mds()
        for i in range(full_rounds_over_2, full_rounds_over_2 + partial_rounds):
            self.apply_ark(i)
            self.apply_s_box(False)
            self.apply_mds()
        for i in range(full_rounds_over_2 + partial_rounds, full_rounds + partial_rounds):
            self.apply_ark(i)
            self.apply_s_box(True)
            self.apply_mds()


    def apply_ark(self, round_number:int):
        for i in range(len(self.state)):
            self.state[i] += self.config.ark[round_number][i]
    def apply_s_box(self,is_full_round:bool):
        # Full rounds apply the S Box (x^alpha) to every element of state
        if is_full_round:
            for i in range((len(self.state))):
                self.state[i] =  self.state[i] ** self.config.alpha
        # Partial rounds apply the S Box (x^alpha) to just the first element of state
        else:
            self.state[0] = self.state[0] ** self.config.alpha

    def apply_mds(self):
        self.state = matrix_vector_product(self.config.mds, self.state).vals
    
    # Absorbs everything in elements, this does not end in an absorbtion.
    def absorb_internal(self, rate_start_index: int,remaining_elements: List[FQ]):
        capacity = self.config.capacity

        # the absorbing limit in each loop is rate
        while(True):
            # if we can finish in this call
            if rate_start_index + len(remaining_elements) <= self.config.rate:
                for i in range(len(remaining_elements)):
                    self.state[capacity + i + rate_start_index] = remaining_elements[i]
                self.mode = DuplexSpongeMode.Absorbing
                self.mode.Absorbing.next_squeeze_index = rate_start_index + len(remaining_elements)
                return # done!
            
            # otherwise absorb (rate - rate_start_index) elements
            num_elements_absorbed = self.config.rate - rate_start_index
            for i in range(num_elements_absorbed):
                self.state[capacity + i + rate_start_index] = remaining_elements[i]
            self.permute()
            remaining_elements = remaining_elements[num_elements_absorbed:]
            rate_start_index = 0
    
    def squeeze_field_elements(self, num_elements:int) -> List[FQ]:
        squeezed_elems = [self.F.zero()]*num_elements  #will be mutated in the squeeze_internal func
        output = []
        match self.mode:
            case DuplexSpongeMode.Absorbing:
                self.permute()
                output = self.squeeze_internal(0, squeezed_elems)

            case DuplexSpongeMode.Squeezing:
                squeeze_index = self.mode.Squeezing.next_squeeze_index
                if squeeze_index == self.config.rate:
                    self.permute()
                    squeeze_index = 0
                output = self.squeeze_internal(squeeze_index, squeezed_elems)   #need to set absorb index back

        return output
    
    # output will be mutated
    def squeeze_internal(self, rate_start_index: int, output: List[FQ]):
        output_remaining = output
        capacity = self.config.capacity
        while(True):
            #if we can finish in this call
            if rate_start_index + len(output_remaining) <= self.config.rate:
                output_remaining = self.state[capacity + rate_start_index: capacity + len(output_remaining) + rate_start_index]
                self.mode = DuplexSpongeMode.Squeezing
                self.mode.Squeezing.next_squeeze_index = rate_start_index + len(output_remaining)
                break
            # otherwise squeeze (rate - rate_start_index) elements
            num_elements_squeezed = self.config.rate - rate_start_index
            output_remaining[:num_elements_squeezed] = self.state[capacity + rate_start_index: capacity + num_elements_squeezed + rate_start_index]

            # Unless we are done with squeezing in this call, permute.
            if len(output_remaining) != self.config.rate:
                self.permute()
            # Repeat with updated output slices
            output_remaining = output_remaining[num_elements_squeezed:]
            rate_start_index = 0
        return output_remaining

def mds_matrix_generator(field_p: FQ, t:int):
    """
    This function generates a maximum distance separable (MDS) matrix,
    which is used in linear layer of Poseidon hush function.

    :param field_p: A field field_p of type galois.GF(p).
    :param int t: The size of Poseidon's inner state
    :return: 2-dim array of size t*t consist of filed elements
    :rtype:
    """
    x_vec = [field_p(ele) for ele in range(0, t)]
    y_vec = [field_p(ele) for ele in range(t, 2 * t)]

    mds_matrix = np.array([[field_p.zero()]*t]*t)
    for i in range(t):
        for j in range(t):
            mds_matrix[i, j] = field_p(1) / (x_vec[i] + y_vec[j])

    return mds_matrix


def init_state_for_grain(alpha, p, prime_bit_len, t, full_round, partial_round) -> List[int]:
    """
    The function generates the initial state for Grain LFSR  in a self-shrinking mode.

    Initialize the state with 80 bits b0, b1, . . . , b79, where

    (a) b0, b1 describe the field,
    (b) bi for 2 ≤ i ≤ 5 describe the S-Box,
    (c) bi for 6 ≤ i ≤ 17 are the binary representation of prime_bit_len,
    (d) bi for 18 ≤ i ≤ 29 are the binary representation of t,
    (e) bi for 30 ≤ i ≤ 39 are the binary representation of RF ,
    (f) bi for 40 ≤ i ≤ 49 are the binary representation of RP , and
    (g) bi for 50 ≤ i ≤ 79 are set to 1.

    :param int alpha: The power of S-box.
    :param int p: The prime field modulus.
    :param int prime_bit_len: The number of bits of the Poseidon prime field modulus.
    :param int t: The size of Poseidon's inner state
    :param int full_round: Number of full rounds
    :param int partial_round: Number of partial rounds
    :return: Initialized state with 80 elements of type int.
    :rtype list:
    """
    init_state = []
    # Choice of encoding for alpha, consistent with filecoin documentation except else
    if alpha == 3:
        exp_flag = 0
    elif alpha == 5:
        exp_flag = 1
    elif alpha == -1:
        exp_flag = 2
    else:
        exp_flag = 3

    init_state += [int(_) for _ in (bin(p % 2)[2:].zfill(2))]
    init_state += [int(_) for _ in (bin(exp_flag)[2:].zfill(4))]
    init_state += [int(_) for _ in (bin(prime_bit_len)[2:].zfill(12))]
    init_state += [int(_) for _ in (bin(t)[2:].zfill(12))]
    init_state += [int(_) for _ in (bin(full_round)[2:].zfill(10))]
    init_state += [int(_) for _ in (bin(partial_round)[2:].zfill(10))]
    init_state += [int(1)] * 30

    return init_state

def calc_round_constants(t, full_round, partial_round, p, field_p, alpha, prime_bit_len) -> List[List[FQ]]:
    """
    This function generates constants for addition at each round.
    From the poseidon paper:
    The round constants are generated using the Grain LFSR [23] in a self-shrinking mode:

    1. Initialize the state with 80 bits b0, b1, . . . , b79 using function init_state_for_grain.
    2. Update the bits using bi+80 = bi+62 ⊕ bi+51 ⊕ bi+38 ⊕ bi+23 ⊕ bi+13 ⊕ bi.
    3. Discard the first 160 bits.
    4. Calculate next bits and state using function calc_next_bits.

    Using this method, the generation of round constants depends on the specific instance, and thus different
    round constants are used even if some of the chosen parameters (e.g., n and t) are the same.

    If a randomly sampled integer is not in Fp, we discard this value and take the next one. Note that
    cryptographically strong randomness is not needed for the round constants, and other methods can also be used.

    :param int t: The size of Poseidon's inner state
    :param int full_round: Number of full rounds
    :param int partial_round: Number of partial rounds
    :param int p: The prime field modulus.
    :param field_p: A field field_p of type galois.GF(p).
    :param int alpha: The power of S-box.
    :param int prime_bit_len: The number of bits of the Poseidon prime field modulus.
    :return: Matrix of field elements of size (full_round + partial_round, t).
        Each t element corresponds to one round constant.
    :rtype list:
    """
    rc_number = t * (full_round + partial_round)

    state = init_state_for_grain(alpha, p, prime_bit_len, t, full_round, partial_round)
    rc_field = []
    # Discard first 160 output bits:
    for _ in range(0, 160):
        new_bit = state[62] ^ state[51] ^ state[38] ^ state[23] ^ state[13] ^ state[0]
        state.pop(0)
        state.append(new_bit)

    while len(rc_field) < rc_number:
        state, bits = calc_next_bits(state, prime_bit_len)

        rc_int = int("".join(str(i) for i in bits), 2)
        if rc_int < p:
            rc_field.append(field_p(rc_int))

    rc_field = np.reshape(rc_field,(full_round + partial_round,t))
    # rc_field = to_FQ_matrix(rc_field, Fp)
    return rc_field

def calc_next_bits(state, prime_bit_len):
    """
    Function generate new LFSR state after shifts new field_size number generated

    - Update the bits using bi+80 = bi+62 ⊕ bi+51 ⊕ bi+38 ⊕ bi+23 ⊕ bi+13 ⊕ bi.
    - Evaluate bits in pairs: If the first bit is a 1, output the second bit. If it is a 0, discard the second bit.

    :param list state: Current LFSR state
    :param int prime_bit_len: The number of bits of the Poseidon prime field modulus.
    :return: New LFSR state after shifts and new field_size number generated.
    :rtype list, list:
    """
    bits = []
    while len(bits) < prime_bit_len:
        new_bit_1 = state[62] ^ state[51] ^ state[38] ^ state[23] ^ state[13] ^ state[0]
        state.pop(0)
        state.append(new_bit_1)

        new_bit_2 = state[62] ^ state[51] ^ state[38] ^ state[23] ^ state[13] ^ state[0]
        state.pop(0)
        state.append(new_bit_2)

        if new_bit_1 == 1:
            bits.append(new_bit_2)

    return state, bits


class Test(unittest.TestCase):
    def test_poseidon_sponge_consistency(self):
        # a= Fp(3)**(1)
        # print("a----->", a)
        sponge_config = PoseidonConfig.test_config()
        # print("self.config.ark---------->", sponge_config.ark.shape())
        sponge = PoseidonSponge(sponge_config, Fp)
        sponge.absorb([Fp(1), Fp(3)])
        vals = sponge.squeeze_field_elements(2)
        assert (vals == [Fp(28455601630849963074343334291022831870584355998338355204089747336059046197), Fp(3858766229754261597253805836148458040231891669234961966896513272669443437983)])


if __name__ == '__main__':
    unittest.main()