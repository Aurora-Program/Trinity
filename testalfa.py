from aurora_core.transcender import Trancender

from alfa2 import * 



A = [1, 0, 1]
B = [1, 1, 0]
M = [0, 1,1 ]  # XOR, XNOR, XOR



def test():
    # Create an instance of Trigate
    trigate = Trigate(A, B, None , M)
    
    # Test operar method
    result_operar = trigate.infer()
    result_learn = trigate.learn()
    result_synthesize = trigate.synthesis()
    
    
    # Test deducir method

    print(f"Ingerir result: {result_operar}")
    print(f"Aprender result: {result_learn}")
    print(f"Sintetizar result: {result_synthesize}")
    duduccion = trigate.deduce_full_state(result_learn, result_synthesize, None, None, result_operar) # Ex
    print(f"Deducir result: {duduccion}")
        
    # Test sintet



def testt():
    t = Trancender()


test()


