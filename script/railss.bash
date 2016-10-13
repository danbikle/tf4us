#!/bin/bash

# railss.bash
# This script should start the webserver.
# The server should listen on port 1840.
# The server should listen on all interfaces.
`dirname $0`/../bin/rails s -p 1840 -b 0.0.0.0

exit

