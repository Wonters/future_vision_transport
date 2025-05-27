#!/bin/bash

# run the app
supervisord -c /app/supervisord.conf

exec "$@"