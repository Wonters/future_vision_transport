#!/bin/bash

# run the app
supervisord -c /etc/supervisor/supervisord.conf

exec "$@"