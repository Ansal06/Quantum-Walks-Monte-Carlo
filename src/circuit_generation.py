# This file contains the functions to build the n-layer biased, unbiased, and hadamard walk circuits
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt
import numpy as np
from qiskit.visualization import plot_histogram
%matplotlib inline


def build_qgb_paperstyle(n_layers: int) -> QuantumCircuit:
    """
    Making sure we measure a qubit at the nth layer once
    no more operations remain to be done on that qubit
    
    """
    num_pos = 2 * n_layers + 1  # position qubits q1...q{2n+1}
    total_qubits = num_pos + 1  # +1 for control q0
    qc = QuantumCircuit(total_qubits, num_pos)
    
    ctrl = 0
    center = num_pos // 2 + 1  # Center position (q5 for n=4)
    
    # Initialize: ball at center
    qc.reset(ctrl)
    qc.x(center)
    
    # Build n_layers pegs - triangle EXPANDS outward
    for layer in range(n_layers):
        qc.h(ctrl)
        
        # Each layer expands: center ±(layer+1)
        left_bound = center - (layer + 1)
        right_bound = center + (layer + 1)
        
        # Generate all CSWAP positions for this layer
        positions = list(range(left_bound, right_bound))
        
        # Apply CSWAP cascade with CX rebalancing
        for i, pos in enumerate(positions):
            qc.cswap(ctrl, pos, pos + 1)
            
            # Add CX rebalancing ONLY if this is NOT the last CSWAP in this layer
            if i < len(positions) - 1:
                qc.cx(pos + 1, ctrl)
        
        # Reset control qubit after each layer (except the last)
        if layer < n_layers - 1:
            qc.reset(ctrl)
    
    # Measure all position qubits (q1 through q{2n+1})
    #measured_qubits = list(range(1, total_qubits))
    #qc.measure(measured_qubits, range(num_pos))
    qc.measure(list(range(1, total_qubits)), list(range(num_pos)))
    
    return qc


def build_biased_qgb(n_layers: int, direction='right'):
    """
    Creates exponential distribution QGB that measures ALL position qubits
    
    """
    num_pos = 2 * n_layers + 1  # 9 position qubits for n_layers=4
    total_qubits = num_pos + 1  # 10 total qubits (q0-q9)
    qc = QuantumCircuit(total_qubits, num_pos)  # 9 classical bits for q1-q9
    
    ctrl = 0
    center = num_pos // 2 + 1  # Center at q5
    
    qc.reset(ctrl)
    qc.x(center)
    
    for layer in range(n_layers):
        if direction == 'right':
            theta = np.pi/4 - layer * np.pi/16
        else:
            theta = 3*np.pi/4 + layer * np.pi/16
            
        qc.rx(theta, ctrl)
        
        left_bound = center - (layer + 1)
        right_bound = center + (layer + 1)
        positions = list(range(left_bound, right_bound))
        
        for i, pos in enumerate(positions):
            qc.cswap(ctrl, pos, pos + 1)
            if i < len(positions) - 1:
                qc.cx(pos + 1, ctrl)
        
        if layer < n_layers - 1:
            qc.reset(ctrl)
    
    position_qubits = list(range(1, total_qubits))  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
    classical_bits = list(range(num_pos))           # [0, 1, 2, 3, 4, 5, 6, 7, 8]
    
    qc.measure(position_qubits, classical_bits)
    
    return qc


def hadamard_walk(n_steps=4):
    # 1 coin qubit + 4 position qubits
    n_pos = 4
    qc = QuantumCircuit(1 + n_pos, n_pos)
    coin = 0
    pos = list(range(1, 1 + n_pos))
    
    qc.x(pos[3])
    
    # prepare coin in (|0> + i|1>)/sqrt(2)
    qc.h(coin)
    qc.s(coin)        # now state = H|0> = (|0>+|1>)/sqrt2, then S gives (|0>+i|1>)/sqrt2
    
    for _ in range(n_steps):
        # standard Hadamard coin
        qc.h(coin)
        
        # conditional shift: move right on |0>, left on |1>
        for i, p in enumerate(pos):
            qc.cry(np.pi/8 * (i+1), coin, p)
        
        # rz phase to sharpen interference
        qc.rz(np.pi/4, coin)
    
    # measure only the position register
    qc.measure(pos, range(n_pos))
    return qc
