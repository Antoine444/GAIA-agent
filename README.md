---
title: agent-GAIA
emoji: 🕵🏻‍♂️
colorFrom: purple
colorTo: green
sdk: gradio
sdk_version: 5.38.0
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 480
license: mit
---

# AI Agent for GAIA Benchmark Questions

## Overview

This project implements an AI agent designed to answer questions from the validation split of the GAIA benchmark. The agent leverages machine learning and natural language processing techniques to understand questions and provide accurate answers based on a structured dataset. This project was done for the completion of the HuggingFace certificate for their AI Agent Course.

## Features

- Processes and answers questions derived from the GAIA benchmark validation set.
- Utilizes a knowledge base for document retrieval and similarity searches to generate precise responses.
- Provides a user-friendly interface to interact with the AI agent for query processing.

## Project Structure

- `agent.py`: Contains the core logic and functionality of the AI agent, including question processing, the tools the agent can use and answer generation.
- `app.py`: Main application script that sets up and runs the service interface, facilitating interaction with the AI agent.
- `requirements.txt`: Lists all dependencies required to run the project, ensuring easy setup and installation.
- `system_prompt.txt`: Contains system prompts that guide the AI agent's response generation, providing context and instructions.
- `supabase_documents.csv`: A CSV file containing document data used by the agent for information retrieval and knowledge base operations.

## Setup and Installation

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or higher
- pip (Python package manager)

### Installation Steps

```bash
git clone https://huggingface.co/spaces/antoine-444/agent-GAIA
cd agent-GAIA

pip install -r requirements.txt
```

You will also need to create API keys for the Supabase and Groq provider.