'''
split - Interface between quantumsim and quantumerrorcorrection.
Written by Brian Tarasinski.

(c) 2017 Brian Tarasinski
Distributed under the GNU GPLv3. See LICENSE.txt or
https://www.gnu.org/licenses/gpl.txt

This code takes in a circuit defined in the quantumsim module, and
splits it into regions over which errors can occur for single qubits.
These then correspond to the weights we put on individual edges in our
syndrome graph.

The function 'make_circuit' generates a circuit as per the Surface-17
model to be implemented by the DiCarlo group. It can be replaced by
another error correcting model if desired.
'''


import quantumsim.circuit
import quantumsim.sparsedm
import quantumsim.photons
import quantumsim.ptm

import numpy as np


def make_circuit(t1=30000, t2=30000, seed=42, t_gate=20, p_x=1e-4, p_yz=5e-4, t_cph_gate=40, 
        static_flux_std=0, t_meas=300, t_cycle=800, readout_error=0.0015, feedback=False):
    surf17 = quantumsim.circuit.Circuit("Surface 17")

    np.random.seed(seed)

    x_bits = ["X%d" % i for i in range(4)]
    z_bits = ["Z%d" % i for i in range(4)]

    measurement_bits = ["M" + b for b in x_bits + z_bits]
    stabilizer_bits = ["S" + b for b in x_bits + z_bits]

    d_bits = ["D%d" % i for i in range(9)]

    quasi_static_flux = {}

    for b, bm, bs in zip(x_bits + z_bits, measurement_bits, stabilizer_bits):
        surf17.add_qubit(b, t1, t2)
        surf17.add_qubit(bm)
        surf17.add_qubit(bs)
        quasi_static_flux[b] = static_flux_std * np.random.randn()

    for b in d_bits:
        surf17.add_qubit(b, t1, t2)
        quasi_static_flux[b] = static_flux_std * np.random.randn()


    def add_x(c, x_anc, d_bits, anc_pulsed, t):
        t += (t_gate + t_cph_gate) / 2
        for d, apulsed in zip(d_bits, anc_pulsed):
            if d is not None:
                c.add_cphase(x_anc, d, time=t)
                if not apulsed:
                    g = c.add_gate("rotate_z", d, angle=quasi_static_flux[
                                   d], time=t + 0.1)
                    g.label = "x"
            if apulsed:
                g = c.add_gate("rotate_z", x_anc, angle=quasi_static_flux[
                               x_anc], time=t + 0.1)
                g.label = "x"
            t += t_cph_gate

    fg_red = ["D0", "D2", "D6", "D8"]
    fg_red_p = ["D1", "D7"]
    fg_purp = ["D3", "D5"]
    fg_purp_p = ["D4"]
    fg_blue = ["X1", "X3"]
    fg_blue_p = ["X0", "X2"]
    fg_green = ["Z2", "Z3"]
    fg_green_p = ["Z0", "Z1"]

    freq_order = {}
    for b in fg_red + fg_red_p:
        freq_order[b] = 2
    for b in fg_purp + fg_purp_p:
        freq_order[b] = 0
    for b in fg_blue + fg_blue_p + fg_green + fg_green_p:
        freq_order[b] = 1

    # the resonator performs a cphase if the first qubit is pulsed and the
    # second is not.
    resonators = [(anc, dat) if freq_order[anc] > freq_order[dat] else (dat, anc) for anc, dats in
                  [
        ("X0", ["D2", "D1"]),
        ("X1", ["D1", "D0", "D4", "D3"]),
        ("X2", ["D5", "D4", "D8", "D7"]),
        ("X3", ["D7", "D6", ]),
        ("Z0", ["D0", "D3", ]),
        ("Z1", ["D2", "D5", "D1", "D4"]),
        ("Z2", ["D4", "D7", "D3", "D6"]),
        ("Z3", ["D5", "D8"]),

    ] for dat in dats]

    flux_dance_x = [
        fg_red_p + fg_blue_p + fg_purp_p,
        fg_red + fg_blue_p + fg_purp,
        fg_red + fg_blue + fg_purp,
        fg_red_p + fg_blue + fg_purp_p]

    flux_dance_z = [
        fg_red + fg_green + fg_purp,
        fg_red_p + fg_green_p + fg_purp_p,
        fg_red_p + fg_green + fg_purp_p,
        fg_red + fg_green_p + fg_purp,
    ]

    t_next_cphase = (t_gate + t_cph_gate) / 2
        
    for slice in flux_dance_x:
        for a, d in resonators:
            if a in x_bits or d in x_bits:
                if a in slice and d not in slice:
                    surf17.add_cphase(a, d, time=t_next_cphase)
        for a in slice:
            g = surf17.add_gate("rotate_z", a, angle=quasi_static_flux[a], time=t_next_cphase + 0.1)
            g.label = 'x'
        t_next_cphase += t_cph_gate

    t2 = m_start = t_gate + 4 * t_cph_gate
    t_next_cphase = t2 + (t_gate + t_cph_gate) / 2

    for slice in flux_dance_z:
        for a, d in resonators:
            if a in z_bits or d in z_bits:
                if a in slice and d not in slice:
                    surf17.add_cphase(a, d, time=t_next_cphase)
        for a in slice:
            g = surf17.add_gate("rotate_z", a, angle=quasi_static_flux[a], time=t_next_cphase + 0.1)
            g.label = 'x'
        t_next_cphase += t_cph_gate
  
     
    sampler = quantumsim.circuit.BiasedSampler(
        readout_error=readout_error, alpha=1, seed=seed)

    for b in d_bits:
        surf17.add_rotate_y(b, angle=np.pi / 2,
                            dephasing_angle=p_yz, dephasing_axis=p_x, time=0)
        surf17.add_rotate_y(b, angle=-np.pi / 2, dephasing_angle=p_yz,
                            dephasing_axis=p_x, time=4 * t_cph_gate + t_gate)

    for b in x_bits:
        surf17.add_rotate_y(b, angle=np.pi / 2,
                            dephasing_angle=p_yz, dephasing_axis=p_x, time=0)
        
        normal_rotation = quantumsim.circuit.RotateY(b, angle=-np.pi / 2, dephasing_angle=p_yz,
                            dephasing_axis=p_x, time=4 * t_cph_gate + t_gate)
        back_rotation =  quantumsim.circuit.RotateY(b, angle=np.pi / 2, dephasing_angle=p_yz,
                            dephasing_axis=p_x, time=4 * t_cph_gate + t_gate)

        if feedback:
            cond_gate = quantumsim.circuit.ConditionalGate(
                    time=4*t_cph_gate+t_gate,
                    control_bit="S"+b,
                    zero_gates=[normal_rotation],
                    one_gates=[back_rotation]
                    )
        else:
            cond_gate = normal_rotation

        surf17.add_gate(cond_gate)

    for b in z_bits:
        surf17.add_rotate_y(b, angle=np.pi / 2,
                            dephasing_angle=p_yz, dephasing_axis=p_x, time=t2)
        normal_rotation = quantumsim.circuit.RotateY(b, angle=-np.pi / 2, dephasing_angle=p_yz,
                            dephasing_axis=p_x, time=t2 + 4 * t_cph_gate + t_gate)
        back_rotation =  quantumsim.circuit.RotateY(b, angle=np.pi / 2, dephasing_angle=p_yz,
                            dephasing_axis=p_x, time=t2+ 4 * t_cph_gate + t_gate)

        if feedback:
            cond_gate = quantumsim.circuit.ConditionalGate(
                    time=t2+4*t_cph_gate+t_gate,
                    control_bit="S"+b,
                    zero_gates=[normal_rotation],
                    one_gates=[back_rotation]
                    )
        else:
            cond_gate = normal_rotation

        surf17.add_gate(cond_gate)


    for b in x_bits:
        m_start = 1.5 * t_gate + 4 * t_cph_gate
        g = quantumsim.circuit.ButterflyGate(
            b, time=m_start, p_exc=0, p_dec=0.005)
        surf17.add_gate(g)
        surf17.add_measurement(b, time=m_start + t_meas, sampler=sampler, output_bit="M"+b)
        surf17.add_gate(quantumsim.circuit.ClassicalCNOT("M"+b, "S"+b, time=m_start+t_meas-10))
        g = quantumsim.circuit.ButterflyGate(
            b, time=m_start + 2 * t_meas, p_exc=0, p_dec=0.015)
        surf17.add_gate(g)

    for b in z_bits:
        m_start = t2 + 1.5 * t_gate + 4 * t_cph_gate
        g = quantumsim.circuit.ButterflyGate(
            b, time=m_start, p_exc=0.0, p_dec=0.005)
        surf17.add_gate(g)
        surf17.add_measurement(b, time=m_start + t_meas, sampler=sampler, output_bit="M"+b)
        surf17.add_gate(quantumsim.circuit.ClassicalCNOT("M"+b, "S"+b, time=m_start+t_meas-10))
        g = quantumsim.circuit.ButterflyGate(
            b, time=m_start + 2 * t_meas, p_exc=0.0, p_dec=0.015)
        surf17.add_gate(g)

    quantumsim.photons.add_waiting_gates_photons(surf17,
                                                 tmin=0, tmax=t_cycle, alpha0=4, kappa=1 / 250, chi=1.3 * 1e-3)

    # surf17.add_waiting_gates(tmin=0, tmax=t_cycle, 
            # idling_gate=quantumsim.circuit.DepolarizingNoise)

    surf17.order()

    return surf17



def error_rate(single_circ, from_basis="Z", to_basis="Z"):
    """
    Given a circuit `single_circ` involving only one qubit,
    evaluate the error rate on the qubit as

    e0 = 1 - prob(Z+ -> Z+)
    e1 = 1 - prob(Z- -> Z-)

    where prob(A -> B) is the probability that a qubit prepared in the
    A state is found in the B state after application of the circuit.

    If `basis` is "X", calculate e0 = 1-prob(X+ -> X+) etc. instead.

    Return (e0, e1).
    """
    bit = single_circ.get_qubit_names()[0]

    sdm = quantumsim.sparsedm.SparseDM([bit])
    if from_basis == "X":
        sdm.hadamard(bit)
    single_circ.apply_to(sdm)
    if to_basis == "X":
        sdm.hadamard(bit)
    sdm.project_measurement(bit, 0)
    fid0 = sdm.trace()
    sdm = quantumsim.sparsedm.SparseDM([bit])
    if from_basis == "X":
        sdm.hadamard(bit)
    sdm.rotate_y(bit, np.pi)
    single_circ.apply_to(sdm)
    sdm.rotate_y(bit, np.pi)
    if to_basis == "X":
        sdm.hadamard(bit)
    sdm.apply_all_pending()
    sdm.project_measurement(bit, 0)
    fid1 = sdm.trace()

    return (1 - fid0, 1 - fid1)


def data_qubit_errors(c, data_qubit):
    """
    Obtain the error rates on a data qubit by the following procedure.

    Split the quantumsim.Circuit `c` into a set of circuits acting only on `data_qubit`, where
    - cphase gates involving `data_qubit` are used as splitting points, and otherwise ignored, and
    - split either by x-ancillas (i.e. a cphase with a qubits whose name starts with "X")
    - or by z-ancillas (start with "Z")

    Then evaluate the error rates of these sub-circuits using `error_rate` in z-basis.

    Return a dictionary {"x_errors": [[anc_start, anc_end, error_rate],...], "z_errors": ...}

    where anc_1, anc_2 are names of ancilla qubits that delimit each segment. ["Z" ancillas for "x_errors" and vice versa!]
    The lists x_errors and z_errors are ordered in time, with the segment that wraps around in time first.
    """

    assert data_qubit in c.get_qubit_names()

    gates_involving_qubit = [
        g for g in c.gates if data_qubit in g.involved_qubits]

    gates_involving_qubit.sort(key=lambda g: g.time)

    x_slices = []
    x_slice = []
    z_slices = []
    z_slice = []
    last_x_ancilla = None
    last_z_ancilla = None

    for g in gates_involving_qubit:
        if g.involved_qubits == [data_qubit]:
            x_slice.append(g)
            z_slice.append(g)
        else:
            other_ancilla = [
                qb_name for qb_name in g.involved_qubits if qb_name != data_qubit][0]
            if other_ancilla[0] == "X":
                # two_qubit gate with an "X" ancilla involved, split here
                x_slices.append([last_x_ancilla, other_ancilla, x_slice])
                x_slice = []
                last_x_ancilla = other_ancilla
            elif other_ancilla[0] == "Z":
                z_slices.append([last_z_ancilla, other_ancilla, z_slice])
                z_slice = []
                last_z_ancilla = other_ancilla
                pass
    # the last slice wraps around to sit in front of the first slice
    x_slices[0][0] = last_x_ancilla
    x_slices[0][2] = x_slice + x_slices[0][2]
    z_slices[0][0] = last_z_ancilla
    z_slices[0][2] = z_slice + z_slices[0][2]

    x_slices_with_error = []
    for a1, a2, gates in x_slices:
        circ = quantumsim.circuit.Circuit()
        circ.add_qubit(data_qubit)
        circ.gates = gates

        e0, e1 = error_rate(circ)

        x_slices_with_error.append((a1, a2, (e0 + e1) / 2))

    z_slices_with_error = []
    for a1, a2, gates in z_slices:
        circ = quantumsim.circuit.Circuit()
        circ.add_qubit(data_qubit)
        circ.gates = gates

        e0, e1 = error_rate(circ)

        z_slices_with_error.append((a1, a2, (e0 + e1) / 2))

    return {
        "z_errors": x_slices_with_error,
        "x_errors": z_slices_with_error
    }


def final_readout_data_qubit_errors(c, data_qubit):
    """
    Obtain the error rates on a data qubit just before a final readout by the following procedure.

    Split the quantumsim.Circuit `c` into a set of circuits acting only on `data_qubit`, where
    - we find the last cphase gate involving `data_qubit` and  an x-ancilla and ignore everything before that,
    - take all single qubit gates till the end of the cycle (where the measurement will take place)

    Then evaluate the error rates of these sub-circuits using `error_rate` from x-basis to z-basis.

    Do the same for the last z-ancilla, and error from z-basis to z-basis.

    Return a dictionary {"x_errors": error_rate_z_to_z, "z_error" error_rate_x_to_z}

    [again, "Z" ancillas for "x_errors" and vice versa!]
    """

    assert data_qubit in c.get_qubit_names()

    gates_involving_qubit = [
        g for g in c.gates if data_qubit in g.involved_qubits]

    gates_involving_qubit.sort(key=lambda g: g.time)

    x_slices = []
    x_slice = []
    z_slices = []
    z_slice = []
    last_x_ancilla = None
    last_z_ancilla = None

    for g in gates_involving_qubit:
        if g.involved_qubits == [data_qubit]:
            x_slice.append(g)
            z_slice.append(g)
        else:
            other_ancilla = [
                qb_name for qb_name in g.involved_qubits if qb_name != data_qubit][0]
            if other_ancilla[0] == "X":
                # two_qubit gate with an "X" ancilla involved, split here
                x_slices.append([last_x_ancilla, other_ancilla, x_slice])
                x_slice = []
                last_x_ancilla = other_ancilla
            elif other_ancilla[0] == "Z":
                z_slices.append([last_z_ancilla, other_ancilla, z_slice])
                z_slice = []
                last_z_ancilla = other_ancilla
                pass

    # now, the last slice is all we care about.

    x_slices[0][0] = last_x_ancilla
    x_slices[0][2] = x_slice + x_slices[0][2]
    z_slices[0][0] = last_z_ancilla
    z_slices[0][2] = z_slice + z_slices[0][2]

    x_gates = x_slice
    z_gates = z_slice

    # x error
    circ = quantumsim.circuit.Circuit()
    circ.add_qubit(data_qubit)
    circ.gates = x_gates
    e0, e1 = error_rate(circ, from_basis="X")
    x_error_to_end = (e0 + e1) / 2

    circ = quantumsim.circuit.Circuit()
    circ.add_qubit(data_qubit)
    circ.gates = z_gates

    e0, e1 = error_rate(circ)

    z_error_to_end = (e0 + e1) / 2

    return {
        "z_errors": x_error_to_end,
        "x_errors": z_error_to_end
    }


def ancilla_qubit_errors(c, ancilla_qubit):
    """
    Obtain the error rates along an ancilla, by the following procedure.

    - From the circuit `c`, extract a circuit involving only `ancilla_qubit` by
      removing all cphase gates and rearranging the circuit to reach from
      measurement to the measurement next round. Use `error_rate` to obtain the
      error rates (e0, e1) in z-basis.  Return as `self_errors`.

    - Obtain the  `propagated_errors` by taking each section between two
      cphases involving `ancilla_qubit` that does not contain the measurement,
      and then evaluate the average error rate e = (e0 + e1)/2 in the x basis
      for each segment.

    Return a dictionary {"self_errors": (e0, e1), "propagated_errors": [([data_0, data_1, {...}], e), ...]},
    where data_0, data_1, ... are the data qubits involved in cphases either before or after each of the segments segment
    (whichever are fewer).
    """

    assert ancilla_qubit in c.get_qubit_names()

    gates_involving_qubit = [
        g for g in c.gates if ancilla_qubit in g.involved_qubits]

    gates_involving_qubit.sort(key=lambda g: g.time)

    measurement_index = [
        g.is_measurement for g in gates_involving_qubit].index(True)

    gates_after_meas = gates_involving_qubit[measurement_index + 1:]
    gates_before_meas = gates_involving_qubit[:measurement_index]

    gates = gates_after_meas + gates_before_meas

    # self errors over the whole run
    gates_without_cphases = [
        g for g in gates if g.involved_qubits == [ancilla_qubit]]

    circ = quantumsim.circuit.Circuit()
    circ.add_qubit(ancilla_qubit)
    circ.gates = gates_without_cphases
    self_errors = error_rate(circ)

    # split by cphase gates
    cphase_slices = []
    current_slice = []
    last_data_qubit = None

    for g in gates:
        if g.involved_qubits == [ancilla_qubit]:
            current_slice.append(g)
        else:
            next_data_qubit = [
                qb_name
                for qb_name in g.involved_qubits
                if qb_name != ancilla_qubit
            ][0]
            # two_qubit gate
            cphase_slices.append(
                [last_data_qubit, next_data_qubit, current_slice])
            current_slice = []
            last_data_qubit = next_data_qubit

    # throw out the first slice as it is not between two ancillas
    cphase_slices = cphase_slices[1:]

    slices_with_error = []
    all_data_so_far = []
    for d1, d2, gts in cphase_slices:
        circ = quantumsim.circuit.Circuit()
        circ.add_qubit(ancilla_qubit)
        circ.gates = gts

        e0, e1 = error_rate(circ, from_basis="X", to_basis="X")

        all_data_so_far.append(d1)
        slices_with_error.append((all_data_so_far.copy(), (e0 + e1) / 2))

    all_data_bits = all_data_so_far + [d2]

    slices_with_error_and_short_data_list = []
    for dbits, er in slices_with_error:
        if len(all_data_bits) < 2 * len(dbits):
            dbits = [b for b in all_data_bits if b not in dbits]

        slices_with_error_and_short_data_list.append((dbits, er))

    return {
        "self_errors": self_errors,
        "propagated_errors": slices_with_error_and_short_data_list}
