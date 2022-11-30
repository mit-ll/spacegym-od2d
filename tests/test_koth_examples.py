# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from orbit_defender2d.king_of_the_hill.examples.core_random_game import run_core_random_game
from orbit_defender2d.king_of_the_hill.examples.pettingzoo_random_game import run_pettingzoo_random_game
from orbit_defender2d.king_of_the_hill.examples.server_random_game import run_server_random_game
from orbit_defender2d.king_of_the_hill.examples.server_2player_random_game import run_server_2player_random_game

def test_core_random_game():
    run_core_random_game()

def test_core_pettingzoo_random_game():
    run_pettingzoo_random_game()

def test_server_random_game():
    run_server_random_game()

def test_server_2player_random_game():
    run_server_2player_random_game()
