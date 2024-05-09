# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [UNRELEASED] - XXXX.XX.XX

## [v0.1.0] - 2024.05.09

### Added

- An AI agent against which a human can play. Run w/
```
conda activate od2d_v0.1.0
cd src/orbit_defender2d/king_of_the_hill/examples   # need to have this as working directory to work properly
python playerCLI_vs_AI.py
```
- Ability to create asymmetric games. See `koth.py`
- Game server that can run a game that two remote users log in to play. Game server can display it's own render of the game
- `koth.py`
    - Ability to create asymmetric games. All game parameters can be asymmetric between players including number of tokens, fuel, ammo, points, win score, etc...
    - Randomizer that can be used to generate random Initial game conditions for use with AI training.
    - More print function options to display info as game is played
    - Logging functions to log game to file

- `orbit_defender2d/king_of_the_hill/examples/`
    - Added player command line interface against trained AI agent example. This is played on a smaller board (4 rings).

### Fixed

### Changed

- Updated some required packages in the environment. Added torch to play trained AI agent.
- Changed some default names in Utils.py. seeker displays as HVA and bludger displays as Patrol. Beta changed to Bravo.
- 'koth.py'
    - Some changes to scoring, added fuel points baed on remaining fuel. Goal Sector points remain cumulative.
    - Made many actions that are not useful illegal (cannot attack already inactive tokens for example). This helps AI training immensely
- `pettingzoo_env.py`
    - Updated the renderer to have larger, easier to see tokens with numbers. Made rendered window resizable

### Removed

## [v0.0.12] - 2024.05.08

### Added

- User Interface
- CHANGELOG.md

### Fixed

### Changed

### Removed

## [v0.0.11] - 2023.02.15