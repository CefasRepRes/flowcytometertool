#!/usr/bin/env python3
"""
Flow Cytometer → Dashboard Sender
Uses flowcytoendpoint.py for ServiceBus config.
"""

import json
import logging
import sys
from azure.servicebus import ServiceBusClient, ServiceBusMessage

import flowcytoendpoint as endpoint


def send_to_dashboard(packet: dict) -> bool:
    """Send a flat QC packet to the dashboard queue."""
    try:
        data = json.dumps(packet, indent=4)

        with ServiceBusClient.from_connection_string(endpoint.connstr) as client:
            with client.get_queue_sender(endpoint.queue_name) as sender:
                message = ServiceBusMessage(data)
                sender.send_messages(message)

        logging.info("Dashboard send succeeded")
        return True

    except Exception as exc:
        logging.exception(f"Dashboard send failed: {exc}")
        return False


def main():
    """Manual test."""
    logging.basicConfig(level=logging.DEBUG)

    test_packet = {
        "version": endpoint.protocol_version,
        "system_serial_no": endpoint.system_serial_no,
        "timestamp": "2026-03-10T12:00:00Z",
        "example": 123
    }

    send_to_dashboard(test_packet)


if __name__ == "__main__":
    main()