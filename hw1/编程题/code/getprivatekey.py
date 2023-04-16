import math
import sys
from Crypto.PublicKey import RSA

arsa = RSA.generate(1024)
arsa.p = 275127860351348928173285174381581152299
arsa.q = 319576316814478949870590164193048041239
arsa.e = 65537
arsa.n = arsa.p * arsa.q
Fn = int((arsa.p - 1) * (arsa.q - 1))
i = 1
while (True):
    x = (Fn * i) + 1
    if (x % arsa.e == 0):
        arsa.d = x / arsa.e
        break
    i = i + 1
private = open('private.pem', 'w')
private.write(arsa.exportKey())
private.close()
# openssl rsa -pubin -text -modulus -in warmup -in pub.pem
