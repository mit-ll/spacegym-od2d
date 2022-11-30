# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# Definitions for orbital gridworld (i.e. the board of the boardgame)

from typing import Tuple, Set
from orbit_defender2d.utils import utils as U

class OrbitGrid:
    def __init__(self, n_rings: int):
        '''create orbital gridworld with n_rings
        
        Args:
            n_rings (int): number of rings in orbit grid
        '''

        assert isinstance(n_rings, int)
        assert n_rings > 0

        self.n_rings = n_rings
        self.n_sectors = int(2**(self.n_rings+1)-1)

    def sector_num2ring(self, sec_num:int) -> int:
        '''get ring number from sector number

        Args:
            sec_num (int): sector number

        Returns:
            ring (int): number or ring sector lies within
        '''
        assert isinstance(sec_num, int)
        assert 0 <= sec_num < self.n_sectors

        return (sec_num + 1).bit_length() - 1


    def sector_num2coord(self, sec_num:int) -> Tuple:
        '''convert sector number representation into radius,azimuth coordinates

        Args:
            sec_num (int): sector number

        Returns:
            sec_coords (Tuple): sector coordinates in (radius, azimuth). 
                radius is ring number and azimuth in sector number within ring
        '''

        assert isinstance(sec_num, int)
        assert 0 <= sec_num < self.n_sectors, "invalid sector number {}. must be in [0,{})".format(sec_num,self.n_sectors)

        ring = self.sector_num2ring(sec_num=sec_num)
        azim = sec_num - 2**(ring) + 1

        return (ring, azim)

    def sector_coord2num(self, ring:int, azim:int) -> int:
        '''convert sector coordinates (ring, azimuth) to sector number

        Args:
            ring (int): ring of sector
            azim (int): loaciton along ring of sector

        Returns:
            (int) sector number

        '''
        assert isinstance(ring, int)
        assert 0 <= ring <= self.n_rings
        assert isinstance(azim, int)
        assert 0 <= azim < 2**ring

        return 2**ring + azim - 1

    def get_num_sectors_in_ring(self, ring:int) -> int:
        ''' given a valid ring number, return number of sectors in that ring
        '''
        assert isinstance(ring, int)
        assert 0 <= ring <= self.n_rings

        return 2**ring

    def get_relative_azimuth_sector(self, sec_num:int, rel_azim:int) -> int:
        ''' get sector in ring with a relative azimuth

        Args:
            sec_num (int): current sector number
            rel_azim (int): relative azimuth to new sector

        Returns:
            new_sec_num (int): sector number of new sector
        '''
        (cur_ring, cur_azim) = self.sector_num2coord(sec_num=sec_num)
        n_sec_in_ring = self.get_num_sectors_in_ring(cur_ring)
        new_azim = (cur_azim + rel_azim) % n_sec_in_ring
        return self.sector_coord2num(cur_ring, new_azim)

    def get_prograde_sector(self, sec_num:int) -> int:
        ''' get sector prograde to given sector

        Args:
            sec_num (int): current sector number

        Returns:
            pro_sec_num (int): prograde sector number
        '''
        return self.get_relative_azimuth_sector(sec_num=sec_num, rel_azim=1)

    def get_retrograde_sector(self, sec_num:int) -> int:
        ''' get sector retrograde to given sector

        Args:
            sec_num (int): current sector number

        Returns:
            ret_sec_num (int): retrograde sector number
        '''
        return self.get_relative_azimuth_sector(sec_num=sec_num, rel_azim=-1)


    def get_radial_out_sector(self, sec_num:int) -> int:
        ''' get radially outward sector from current sector
        Note: always returns the lower-value of the two outer sectors

        Args:
            sec_num (int): current sector number

        Returns:
            out_sec_num (int): sector number of outer sector. None if on outer ring
        '''

        (cur_ring, cur_azim) = self.sector_num2coord(sec_num=sec_num)

        # check for outermost ring, return None
        if cur_ring >= self.n_rings:
            return None

        # compute outer ring number
        out_ring = cur_ring + 1

        # computer outer azimuth number
        # always the lower of the two branches in outer ring
        out_azim_bits = U.int2bitlist(cur_azim)
        out_azim_bits.append(0)
        out_azim = U.bitlist2int(out_azim_bits)

        # get outer sector number
        return self.sector_coord2num(out_ring, out_azim)

    def get_radial_in_sector(self, sec_num:int) -> int:
        ''' get radially inward sector from current sector
        Note: always returns the lower-value of the two outer sectors

        Args:
            sec_num (int): current sector number

        Returns:
            out_sec_num (int): sector number of outer sector. None if on outer ring
        '''

        (cur_ring, cur_azim) = self.sector_num2coord(sec_num=sec_num)

        # check for innermost ring, return None
        if cur_ring < 1:
            return None

        # compute outer ring number
        in_ring = cur_ring - 1

        # computer outer azimuth number
        in_azim_bits = U.int2bitlist(cur_azim)[:-1]
        in_azim_bits = [0] if len(in_azim_bits) == 0 else in_azim_bits
        in_azim = U.bitlist2int(in_azim_bits)

        # get outer sector number
        return self.sector_coord2num(in_ring, in_azim)

    def get_all_adjacent_sectors(self, sec_num:int) -> Set:
        ''' get list of sectors adjacent to given sector (including both radial out)

        Args:
            sec_num (int): current sector number

        Returns:
            adj_sec_num_set (set): set of sector numbers adjacent to sec_num
        '''
        pro_sec = self.get_prograde_sector(sec_num)
        ret_sec = self.get_retrograde_sector(sec_num)
        in_sec = self.get_radial_in_sector(sec_num)
        out_sec = self.get_radial_out_sector(sec_num)
        
        # add pro and ret sec, checking for uniqueness
        adj_sec_num_set = set([pro_sec, ret_sec])

        # add radial in sector, checking for existence
        if in_sec is not None:
            adj_sec_num_set.add(in_sec)

        # add both radial out sectors, if valid
        if out_sec is not None:
            adj_sec_num_set.add(out_sec)
            adj_sec_num_set.add(self.get_prograde_sector(out_sec))

        return adj_sec_num_set
        
