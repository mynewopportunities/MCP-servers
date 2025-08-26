# Conversational AI for MSP Customer Support

## Overview
AI-powered chatbots and virtual assistants using Hugging Face models for automated IT support.

## Features
- Password reset automation
- Network troubleshooting
- Account management
- Ticket triage and routing

## Models Used
- microsoft/DialoGPT-large
- facebook/blenderbot-400M-distill
- microsoft/conversational-ai

## Integration Points
- ServiceNow
- Zendesk
- Freshdesk
- ConnectWise PSA

## Quick Start
```bash
cd customer-support/conversational-ai
pip install -r requirements.txt
python mcp_server.py
```