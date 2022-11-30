# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

"""
Clone client Model Two


Ref: https://zguide.zeromq.org/docs/chapter5/#Getting-an-Out-of-Band-Snapshot
Author: Min RK <benjaminrk@gmail.com>

"""

import time
import zmq

from kvsimple import KVMsg

def main():

    # prepare our context and subscriber
    ctx = zmq.Context()
    snapshot = ctx.socket(zmq.DEALER)
    snapshot.linger = 0     # how long messages that have yet to be sent to a peer will linger in memory after disconnect
    snapshot.connect("tcp://localhost:5556")
    subscriber = ctx.socket(zmq.SUB)
    subscriber.linger = 0
    # must set a subscription, missing this step is a common mistake. 
    # https://zguide.zeromq.org/docs/chapter1/#Getting-the-Message-Out
    subscriber.setsockopt_string(zmq.SUBSCRIBE, '') 
    subscriber.connect("tcp://localhost:5557")

    kvmap = {}

    # Get state snapshot
    sequence = 0
    snapshot.send_string("ICANHAZ?")
    while True:
        # print("~~~DEBUG~~~: Attempting to receive snapshot")
        try:
            kvmsg = KVMsg.recv(snapshot)
            print("snapshot RECV~~~\nkey: {}\nbody: {}\nseq: {}".format(kvmsg.key, kvmsg.body, kvmsg.sequence))
        except Exception as ex:
            print("snapshot INTERRUPTED")
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            break

        if kvmsg.key == "KTHXBAI":
            sequence = kvmsg.sequence
            print("Received snapshot=%d" % sequence)
            print("snapshot DONE")
            break   # done
        kvmsg.store(kvmap)

    # Now apply pending updates, discard out-of-sequence messages
    print("\nReceiving on subscriber...\n")
    while True:
        # print("~~~DEBUG~~~: Attempting to receive subscriber")
        try:
            kvmsg = KVMsg.recv(subscriber)
            # print("subscriber RECV~~~\nkey: {}\nbody: {}\nseq: {}".format(kvmsg.key, kvmsg.body, kvmsg.sequence))
        except Exception as ex:
            print("subscriber INTERRUPTED")
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            break
        
        if kvmsg.sequence > sequence:
            sequence = kvmsg.sequence
            kvmsg.store(kvmap)

if __name__ == "__main__":
    main()
