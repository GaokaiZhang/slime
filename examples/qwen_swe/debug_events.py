#!/usr/bin/env python3
"""Debug qwen-code event structure."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# Sample raw output from a successful rollout (for debugging)
sample_events = """
{"type":"system","subtype":"init","uuid":"abc","session_id":"abc","cwd":"/testbed","tools":["task"]}
{"type":"message","message":{"role":"user","parts":[{"text":"Fix the bug"}]}}
{"type":"message","message":{"role":"model","parts":[{"text":"I'll analyze"},{"functionCall":{"name":"read_file","args":{"path":"test.py"}}}]}}
{"type":"message","message":{"role":"user","parts":[{"functionResponse":{"name":"read_file","response":{"result":"file content here"}}}]}}
{"type":"message","message":{"role":"model","parts":[{"text":"I see the issue. Let me fix it."}]}}
"""

events = []
for line in sample_events.strip().split("\n"):
    if line.strip():
        events.append(json.loads(line))

print("=== Parsed Events ===")
for i, event in enumerate(events):
    print(f"\nEvent {i}:")
    print(f"  type: {event.get('type')}")
    if event.get("message"):
        msg = event["message"]
        print(f"  message.role: {msg.get('role')}")
        parts = msg.get("parts", [])
        for j, part in enumerate(parts):
            if isinstance(part, str):
                print(f"    part[{j}]: text string")
            elif isinstance(part, dict):
                if "text" in part:
                    print(f"    part[{j}]: text dict")
                elif "functionCall" in part:
                    print(f"    part[{j}]: functionCall -> {part['functionCall'].get('name')}")
                elif "functionResponse" in part:
                    print(f"    part[{j}]: functionResponse -> {part['functionResponse'].get('name')}")

print("\n=== Testing build_messages_from_events ===")
from examples.qwen_swe.rollout import build_messages_from_events

messages = build_messages_from_events(events)
print(f"\nConverted {len(messages)} messages:")
for i, msg in enumerate(messages):
    print(f"  [{i}] role={msg['role']}: {msg['content'][:50]}...")
