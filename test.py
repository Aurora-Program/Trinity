from Trinity import * 

# Test Trigate class
def test_Operar():
    # Create an instance of Trigate
    trigate = Trigate()
    a=trigate.operar(3, 2, 'XOR')
    print(a)

def test_Aprender():
    # Create an instance of Trigate
    trigate = Trigate()
    a=trigate.aprendizaje(5, 3, 1)
    print(a)


def test_deducir():
    # Create an instance of Trigate
    trigate = Trigate()
    a=trigate.deduccion_inversa('OR', 3, 1)
    print(a)

test_Aprender()
test_Operar()   
test_deducir()




