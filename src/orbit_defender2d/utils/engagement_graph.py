# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# M-v-N engagements between pieces represented as a directed graph

import networkx as nx
import numpy as np
import orbit_defender2d.utils.utils as U

class EngagementGraph:

    def __init__(self, engagements):
        ''' Instantiate engagement graph with game pieces as nodes and attacks/defense as edges

        Args:
            engagements (dict): key is piece id, one for each piece in game
                                value is the piece's engagement tuple (engagement_type, target_piece_id, probability)
        '''

        # Instantiate directional engagement graph
        self.egraph = nx.DiGraph()

        # create nodes from game piece objects
        self.egraph.add_nodes_from(engagements.keys())

        # populate edges based on engagements
        for piece, eng_tuple in engagements.items():

            # assert len(eng_tuple) == 3
            eng_type  = eng_tuple.action_type
            eng_target = eng_tuple.target
            eng_prob = eng_tuple.prob

            # TODO: check that targeted piece exists to ensure erroneous nodes aren't silently added

            if eng_type == U.NOOP:
                # enforce self-target for tuple consistency, no edge to add for no-op
                assert piece == eng_target
            elif eng_type in [U.SHOOT, U.COLLIDE]:
                # check that target is an opponent and add edge
                assert piece.split(':')[0] != eng_target.split(':')[0]
                self.egraph.add_edge(piece, eng_target, etype=eng_type, prob=eng_prob)
            elif eng_type == U.GUARD:
                # check that target is an ally piece, but not a self-guard, and add edge
                assert (piece.split(':')[0] == eng_target.split(':')[0]) and (piece != eng_target)
                self.egraph.add_edge(piece, eng_target, etype=eng_type, prob=eng_prob)
            else:
                raise ValueError("Unrecognized engagement type {}".format(eng_type))

    def resolve_guard_engagements(self):
        ''' evaluate success of guard engagments
            re-routing edges to guardian pieces for successful guards
        Returns:
            guard_outcomes (GuardOutcomesTuple): List of named tuples describing outcomes
        '''

        # keep list of guard action outcomes
        guard_outcomes = []

        # extract guard edges
        guard_edges = [(src, tar, dat) for src, tar, dat in \
            self.egraph.edges(data=True) if \
            dat['etype']==U.GUARD]

        # randomize order for edge evaluation
        np.random.shuffle(guard_edges)

        # iterate through guard edges in random order
        for ge in guard_edges:

            # separate out the guardian, guarded, and edge type data
            grd_src, grd_tar, grd_dat = ge
            guard_prob = grd_dat['prob']

            # ensure valid guard probability
            assert 0 <= guard_prob <= 1.0     

            # remove guard edge because it is being evaluated, 
            # regardless of success of guarding success
            self.egraph.remove_edge(grd_src, grd_tar)

            # collect all incident attacking edges directed at guarded piece
            in_att_edges = [(src, tar, dat) for src, tar, dat in \
                self.egraph.in_edges(grd_tar, data=True) if \
                dat['etype'] in [U.SHOOT, U.COLLIDE]]

            # check if any incident attack edges exist
            # if not, record a trivial engagement outcome
            if len(in_att_edges) == 0:

                # add trivial outcome to sequential list
                guard_outcomes.append(U.EngagementOutcomeTuple(
                    action_type=U.GUARD,
                    attacker=None,
                    target=grd_tar,  
                    guardian=grd_src, 
                    prob=guard_prob, 
                    success=False))

            else:
                guard_count = 0
                # randomize order for edge evaluation
                np.random.shuffle(in_att_edges)

                # iterate through incident attack edges in random order,
                # evaluating guarding success with decaying probability
                decayed_guard_prob = guard_prob
                for iae in in_att_edges:
                    att_src, att_tar, att_dat = iae
                    success = False
                    assert grd_tar == att_tar
                    if np.random.rand() < decayed_guard_prob:
                        success = True
                        self.egraph.remove_edge(att_src, att_tar)
                        self.egraph.add_edges_from([(att_src, grd_src, att_dat)])
                    
                    # add outcome to sequential list
                    guard_outcomes.append(U.EngagementOutcomeTuple(
                        action_type=U.GUARD,
                        attacker=att_src,
                        target=grd_tar,  
                        guardian=grd_src, 
                        prob=decayed_guard_prob, 
                        success=success))
                    guard_count += 1
                    decayed_guard_prob = guard_prob * (0.5**guard_count)

        return guard_outcomes
        
    def resolve_shoot_engagements(self):
        ''' evaluate success of shooting engagments
            removing nodes of successful attacks
        Returns:
            shoot_outcomes (ShootOutcomesTuple): List of named tuples describing outcomes
        '''

        # keep list of guard action outcomes
        shoot_outcomes = []

        # extract shoot edges
        shoot_edges = [(src, tar, dat) for src, tar, dat in \
            self.egraph.edges(data=True) if \
            dat['etype']==U.SHOOT]

        # randomize order for edge evaluation 
        # (although evaluation order shouldn't matter for shooting
        # attacks)
        np.random.shuffle(shoot_edges)

        # establish removal list
        node_removal_set = set()

        # iterate through each shot edge
        for sht_edge in shoot_edges:
            sht_src, sht_tar, sht_dat = sht_edge

            # remove attack edge because it is being evaluated,
            # regardless whether the attack is successful or not
            self.egraph.remove_edge(sht_src, sht_tar)

            # evaluate success of attack
            success = False
            if np.random.rand() < sht_dat['prob']:
                # successful attack, queue target node for removal
                # but don't remove yet incase target has also 
                # launched a shoot attack that has yet to 
                # be evaluated
                success = True
                node_removal_set.add(sht_tar)

            # add outcome to sequential list
            # NOTE: no guardian because guadian engagements resolved first
            shoot_outcomes.append(U.EngagementOutcomeTuple(
                action_type=U.SHOOT,
                attacker=sht_src,
                target=sht_tar,
                guardian='',
                prob=sht_dat['prob'],
                success=success))
        
        # remove successfully shoot attacked nodes simultaneously
        self.egraph.remove_nodes_from(node_removal_set)

        return shoot_outcomes

    def resolve_collide_engagements(self):
        ''' evaluate success of collision engagments
            removing nodes of successful attacks
        Returns:
            collide_outcomes (CollideOutcomesTuple): List of named tuples describing outcomes
        '''

        collide_outcomes = []
        while True:

            # extract current set of collision edges
            collide_edges = [(src, tar, dat) for src, tar, dat in \
                self.egraph.edges(data=True) if \
                dat['etype']==U.COLLIDE]

            # if no more collision edges exist, break
            if len(collide_edges) == 0:
                break

            # randomize order for edge evaluation 
            np.random.shuffle(collide_edges)

            # evaluate first edge
            col_src, col_tar, col_dat = collide_edges[0]

            # remove attack edge because it is being evaluated,
            # regardless whether the attack is successful or not
            self.egraph.remove_edge(col_src, col_tar)

            # evaluate success of attack
            success = False
            if np.random.rand() < col_dat['prob']:
                # successful attack, remove both nodes
                success = True
                self.egraph.remove_nodes_from([col_src, col_tar])

            # add outcome to sequential list
            # NOTE: no guardian because guadian engagements resolved first
            collide_outcomes.append(U.EngagementOutcomeTuple(
                action_type=U.COLLIDE,
                attacker=col_src,
                target=col_tar,
                guardian='',
                prob=col_dat['prob'],
                success=success))

        return collide_outcomes


