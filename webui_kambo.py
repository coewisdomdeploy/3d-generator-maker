#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Kaboomi 3D Studio - Launcher Específico para Kaboomi Party (Godot 4)
Ejecuta la WebUI en modo 'kambo' con rutas y biblioteca directa del juego.
==============================================================================
"""
import os
import sys

if __name__ == "__main__":
    os.environ["STUDIO_PROFILE"] = "kambo"
    if "--mode" not in sys.argv and "--profile" not in sys.argv:
        sys.argv.extend(["--mode", "kambo"])
    import webui
