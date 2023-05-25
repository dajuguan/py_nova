from Cryptodome import Random
from Cryptodome.Random import random
from Cryptodome.Util import number
import unittest

class verifier:
    def setup(self, security):
        # Pick p, q primes such that p | q - 1, that is equvalent to
        # say that q = r*p + 1 for some r
        p = number.getPrime(security, Random.new().read)
        print("p = ",p)
        
        r = 1
        while True:
            q = r*p + 1
            if number.isPrime(q):
                print("q = ",q)
                break
            r += 1
        
        # Compute elements of G = {i^r mod q | i in Z_q*}
        G = [] 
        for i in range(1, q): # Z_q*
            G.append(i**r % q)

        G = list(set(G))
        print("Order of G = {i^r mod q | i in Z_q*} is " + str(len(G)) + " (must be equal to p).")
        
        # Since the order of G is prime, any element of G except 1 is a generator
        g = random.choice(list(filter(lambda e: e != 1, G)))
        print("g = ",g)
                
        h = random.choice(list(filter(lambda e: e != 1 and e != g, G)))
        print("h = ",h)
        
        # g and h are elements of G such that nobody knows math.log(h, g) (log of h base g)
           
        return q,g,h

    def open(self, param, c, m, *r):
        q, g, h = param

        rSum = 0
        for rEl in r:
            rSum += rEl
       
        return c == (pow(g,m,q) * pow(h,rSum,q)) % q  

    def add(self, param, *c):
        q = param[0]
        
        cSum = 1
        for cEl in c:
            cSum *= cEl
        return cSum % q
        
class prover: 
    def commit(self, param, m):
        q, g, h = param
        
        r = number.getRandomRange(1, q-1)
        c = (pow(g,m,q) * pow(h,r,q)) % q
        return c, r

class Test(unittest.TestCase):
    def test(self):
        security = 10
        m1 = 51
        m2 = 63

        v = verifier()
        p = prover()

        param = v.setup(security)

        c1, r1 = p.commit(param, m1)
        c2, r2 = p.commit(param, m2)

        addCommitment = v.add(param, c1, c2)

        print("\nm1:",m1)
        print("m2:",m2)

        print("c1,r1:", c1, ",", r1)
        print("c2,r2:", c2, ",", r2)
        print("Let's multiply c1*c2 in order to get a commitment of m1 + m2.")
        print("c1*c2:", addCommitment) # mod q

        print("\nDoes c1 open to m1?", v.open(param, c1, m1, r1))
        print("Does c2 open to m2?", v.open(param, c2, m2 , r2))

        print("Does c1*c2 open to m1 + m2?", v.open(param,addCommitment, m1 + m2 , r1, r2))

if __name__ == '__main__':
    unittest.main()