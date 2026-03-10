#!/usr/bin/env python3

# ******************************************************************************
# Each PI has a unique serial number
# ******************************************************************************

system_serial_no = "flowcytometer01"

# ******************************************************************************
# Version stamp the dashboard protocol
# ******************************************************************************

protocol_version = "0.0.3"

# ******************************************************************************
# Add your Service Bus configuration parameters here.
# ******************************************************************************

# connstr = os.environ["SERVICE_BUS_CONNECTION_STR"]
# queue_name = os.environ["SERVICE_BUS_QUEUE_NAME"]

connstr = "Endpoint=sb://cit-rd-plankton-svcbus.servicebus.windows.net/;SharedAccessKeyName=key_name;SharedAccessKey=your_key_here="
queue_name = "rv-dashboard"