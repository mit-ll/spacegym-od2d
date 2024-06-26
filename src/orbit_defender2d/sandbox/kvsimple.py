# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

"""
=====================================================================
kvsimple - simple key-value message class for example applications

Author: Min RK <benjaminrk@gmail.com>

"""

import struct # for packing integers
import sys

import zmq

class KVMsg(object):
    """
    Message is formatted on wire as 3 frames:
    frame 0: key (0MQ string)
    frame 1: sequence (8 bytes, network order)
    frame 2: body (blob)
    """
    key = None # key (string)
    sequence = 0 # int
    body = None # blob

    def __init__(self, sequence, key=None, body=None):
        assert isinstance(sequence, int)
        self.sequence = sequence
        self.key = key
        self.body = body

    def store(self, dikt):
        """Store me in a dict if I have anything to store"""
        # this seems weird to check, but it's what the C example does
        if self.key is not None and self.body is not None:
            dikt[self.key] = self

    def send(self, socket):
        """Send key-value message to socket; any empty frames are sent as such."""
        key = '' if self.key is None else self.key
        b_key = key.encode('utf-8') # strings must be converted to bytes: https://github.com/booksbyus/zguide/issues/747
        seq_s = struct.pack('!l', self.sequence)
        body = '' if self.body is None else self.body
        b_body = body.encode('utf-8')
        socket.send_multipart([ b_key, seq_s, b_body ])

    @classmethod
    def recv(cls, socket):
        """Reads key-value message from socket, returns new kvmsg instance."""
        key, seq_s, body = socket.recv_multipart()
        key = key if key else None
        if isinstance(key, bytes):
            key = key.decode('utf-8') # strings are sent as bytes, must decode to strings: https://github.com/booksbyus/zguide/issues/747
        seq = struct.unpack('!l',seq_s)[0]
        body = body if body else None
        if isinstance(body, bytes):
            body = body.decode('utf-8')
        return cls(seq, key=key, body=body)

    def dump(self):
        if self.body is None:
            size = 0
            data='NULL'
        else:
            size = len(self.body)
            data=repr(self.body)
        print("[seq:{seq}][key:{key}][size:{size}] {data}".format(
            seq=self.sequence,
            key=self.key,
            size=size,
            data=data), 
            file=sys.stderr)

# ---------------------------------------------------------------------
# Runs self test of class

def test_kvmsg (verbose):
    print(" * kvmsg: ")

    # Prepare our context and sockets
    ctx = zmq.Context()
    output = ctx.socket(zmq.DEALER)
    output.bind("ipc://kvmsg_selftest.ipc")
    input = ctx.socket(zmq.DEALER)
    input.connect("ipc://kvmsg_selftest.ipc")

    kvmap = {}
    # Test send and receive of simple message
    kvmsg = KVMsg(1)
    kvmsg.key = "key"
    kvmsg.body = "body"
    if verbose:
        kvmsg.dump()
    kvmsg.send(output)
    kvmsg.store(kvmap)

    kvmsg2 = KVMsg.recv(input)
    if verbose:
        kvmsg2.dump()
    assert kvmsg2.key == "key"
    kvmsg2.store(kvmap)

    assert len(kvmap) == 1 # shouldn't be different

    print("OK")

if __name__ == '__main__':
    test_kvmsg('-v' in sys.argv)
