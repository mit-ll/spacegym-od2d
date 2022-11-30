# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# a stand-in ZMQ client to stand-in for unity

import zmq

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

#  Do 10 requests, waiting each time for a response
for request in range(3):
    print(f"Sending request {request} …")
    action_selection = {'alpha':request, 'beta':request+1}
    socket.send_json(action_selection)

    #  Get the reply.
    message = socket.recv_json()
    print(f"Received reply {request} [ {message} ]")
