# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# Definitions for satellite objects that act playing pieces in orbit grid

class Satellite:
    def __init__(self, fuel:float, ammo:int):
        '''
        Args:
            fuel (float): amount of fuel satellite contains
        '''
        self.fuel = fuel
        self.ammo = ammo
