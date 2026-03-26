# PHI De-Identification Pipeline for Clinical Text

## Overview

This project implements a prototype pipeline for de-identifying protected health information (PHI) in free-text clinical notes.

The pipeline:
1. detects standard PHI entities with Microsoft Presidio.
2. de-identifies found entities and keeps pharmacy related terms for further data downstream.

Installation for macos/linux

START VIRTUAL ENVIRONMENT

    python3 -m venv .venv
    source .venv/bin/activate

DOWNLOAD DEPENDENCIES 

    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r requirements.txt
    python -m spacy download en_core_web_sm

RUN CODE 

    python main.py

Installation for windows

START VIRTUAL ENVIRONMENT

    python -m venv .venv
    .venv\Scripts\Activate.ps1

UPGRADE PIP AND INSTALL DEPENDENCIES

    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r requirements.txt
    python -m spacy download en_core_web_sm

RUN CODE 

    python main.py

if any problems occured during runtime, please refer to microsoft manual on how to install presidio.
https://microsoft.github.io/presidio/installation/
