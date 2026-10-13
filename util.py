from pylcp import QuadrupoleMagneticField
from pylcp.atom.dataframe import DataFrame, PredefinedAtom


def make_rubidium_frame(det, s, alpha):
    frame = DataFrame.create_from_atom(PredefinedAtom.Rb87)
    g = frame.get_state_label(0)
    e1 = frame.get_state_label(1)
    e2 = frame.get_state_label(2)
    # TODO: decide if we want to remove F' = 0,1, gotta code the same in pylcp if so
    frame.add_conventional_MOT_lasers(g, e2, 1, 2, 0, 0.01 * s, delta_in_Gamma=True)
    frame.add_conventional_MOT_lasers(g, e2, 2, 3, det, s, delta_in_Gamma=True)
    frame.add_laser_display(0, 2, 3, excited_label=e2)
    frame.add_laser_display(1, 1, 2, excited_label=e2)
    frame.set_magnetic_field(QuadrupoleMagneticField(alpha))
    return frame
