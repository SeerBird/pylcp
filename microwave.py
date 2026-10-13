import pylcp
from util import make_rubidium_frame

frame = make_rubidium_frame(-2,1,1)
ham = frame._hamiltonian()
#ham.add_mu_q_block("g","e1", )
lasers = frame._lasers()
field = frame.magnetic_field
rateeq = pylcp.RateEq(lasers,field,ham)

