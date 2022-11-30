# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

'''
Ref: https://zguide.zeromq.org/docs/chapter5/#Getting-an-Out-of-Band-Snapshot

Clone server Model Two

Author: Min RK <benjaminrk@gmail.com>
'''

import random
import threading
import time

import zmq

from kvsimple import KVMsg
from zhelpers import zpipe

def main():

    # prepare our context and publisher content
    ctx = zmq.Context()
    publisher = ctx.socket(zmq.PUB)
    publisher.bind("tcp://*:5557")

    # set of PAIR sockets for talking to threads (??)
    updates, peer = zpipe(ctx)

    manager_thread = threading.Thread(target=state_manager, args=(ctx,peer))
    manager_thread.daemon = True
    manager_thread.start()

    sequence = 0
    random.seed(time.time())

    try:
        while True: 
            # Distribute as key-value message
            sequence += 1
            kvmsg = KVMsg(sequence)
            kvmsg.key = "%d" % random.randint(1, 10000)
            kvmsg.body = "%d" % random.randint(1, 1000000)
            kvmsg.send(publisher)
            kvmsg.send(updates)
    except KeyboardInterrupt:
        print("Interrupted\n%d messages out" % sequence)

class Route:
    '''A simple struct for routing information for a key-value snapshot'''
    def __init__(self, socket, identity):
        self.socket = socket # ROUTER socket to send to
        self.identity = identity # identity of peer who requested state

def send_single(key, kvmsg, route):
    '''Send one state snapshot key-value pair to a socket

    Hash item table is our kvmsg object, ready to send
    '''
    # send identity of recipient first
    route.socket.send(route.identity, zmq.SNDMORE)
    kvmsg.send(route.socket)

def state_manager(ctx, pipe):
    '''This thread maintains the state and handles requests from clients for snapshots.
    '''
    kvmap = {}
    pipe.send_string("READY")
    snapshot = ctx.socket(zmq.ROUTER)
    snapshot.bind("tcp://*:5556")

    # I *think* the need for the Poller object is 
    # eliminated by using tornado ioloop
    # See: https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/multisocket/tornadoeventloop.html
    poller = zmq.Poller()
    poller.register(pipe, zmq.POLLIN)
    poller.register(snapshot, zmq.POLLIN)

    sequence = 0    # current snapshot version number
    while True:
        try:
            items = dict(poller.poll())
        except (zmq.ZMQError, KeyboardInterrupt):
            break # interrupt/context shutdown

        # Apply state update from main thread
        if pipe in items:
            kvmsg = KVMsg.recv(pipe)
            sequence = kvmsg.sequence
            kvmsg.store(kvmap)

        # Execute state snapshot request
        if snapshot in items:
            msg = snapshot.recv_multipart()
            identity = msg[0]
            request = msg[1].decode('utf-8')    # strings are sent as bytes, decode on receive
            print("~~~DEBUG~~~\nidentity: {}\nrequest: {}".format(identity, request))
            if request == "ICANHAZ?":
                pass
            else:
                print("E: bad request, aborting\n")
                break

            # send state snapshot to client
            route = Route(snapshot, identity)

            # for each entry in kvmap, send kvmsg to client
            for k, v in kvmap.items():
                # print("~~~DEBUG~~~\nkvmap key: {}\nkvmsg key: {}\nkvmsg body: {}\nkvmsg seq: {}".format(k, v.key, v.body, v.sequence))
                send_single(k,v,route)

            # Now send END message with sequence number
            print("Sending state snapshot=%d\n" % sequence)
            snapshot.send(identity, zmq.SNDMORE)
            kvmsg = KVMsg(sequence)
            kvmsg.key = "KTHXBAI"
            kvmsg.body = ""
            kvmsg.send(snapshot)

if __name__ == "__main__":
    main()
    

