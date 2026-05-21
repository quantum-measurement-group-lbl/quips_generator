"""
Models a gas collision with the bead
Treated as instantaneous momentum transfer according to Tsang et al. paper
Integrated into gaussian.py but currently only as a single collision at t/2 --> will need to add a collision frequency model as well here

Also need to addSpecular vs diffuse collision models based off of alpha
    (at the moment, we're only modeling average collision impulses as p ~ sqrt(m_g kB T_g))

Need to add the collisions rates to model

Note that these are calculated in SI units so converted to dimensionless units in gaussian.py
The repository converts them back to SI units at the very end when hosting the server
"""

import numpy as np

# Physical constants
kB = 1.380649e-23  # J/K
u = 1.66053906660e-27  # kg (atomic mass unit)
Tg = 293  # Kelvin  # Taken from Tseng paper

# Gas library (in atomic mass u)
# Feel free to add any other gas types
GAS_SPECIES = {
    "Kr": 83.798,
    "Xe": 131.293,
    "SF6": 146.055,
    "N2": 28.014,
    "H2": 2.016,
}


# Collision where gas bounces off the particle fully (probability 1 - alpha)
# Returns the momentum transfer magnitude after the collision [kg m/s]
def sample_specular_collision(gas: str = "Xe", Tg: float = 293.0):
    pass


# Collision where gas diffuses away from the particle (probability alpha)
# Returns the dimensionless momentum transfer after the collision
def sample_diffuse_collision(gas: str = "Xe", Tg: float = 293.0):
    pass


def average_collision(gas: str = "Xe", Tg: float = 293.0) -> float:
    mg = GAS_SPECIES[gas] * u  # mass of gas in kg
    return np.sqrt(mg * kB * Tg)  # in kg m/s


def apply_kick(pc: float, delta_p: float) -> float:
    # Apply an instantaneous momentum kick to the bead -- from Tseng paper, gas collisions modeled as instantaneous p trasfers
    # No change in position
    return pc + delta_p
