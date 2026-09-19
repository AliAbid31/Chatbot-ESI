"""
app.py : Serveur haute performance CISSOU pour Hugging Face Spaces (16 Go de RAM).
Conserve 100% de l'interface originale static/index.html.
"""
from __future__ import annotations

import os
# L'import de gradio valide les prérequis de Hugging Face Spaces
import gradio as gr 
from cissou import create_app

print("==================================================")
print("Démarrage de CISSOU sur Hugging Face (16 Go RAM)...")
print("==================================================")

# Initialisation de votre application Flask et du moteur de recherche
app = create_app()

if __name__ == "__main__":
    # Hugging Face Spaces écoute obligatoirement sur le port 7860
    port = int(os.environ.get("PORT", 7860))
    print(f"--> CISSOU écoute sur le port {port}")
    
    # threaded=True permet à plusieurs étudiants de poser des questions en même temps
    app.run(host="0.0.0.0", port=port, threaded=True)
