from Trinity import * 

# Test Trigate class
def test_trigate(): 
    # Create an instance of Trigate
    trigate = Trigate()

    # Test the default state
    assert trigate.state == "off", "Default state should be 'off'"

    # Test turning on the trigate
    trigate.turn_on()
    assert trigate.state == "on", "State should be 'on' after turning on"

    # Test turning off the trigate
    trigate.turn_off()
    assert trigate.state == "off", "State should be 'off' after turning off"