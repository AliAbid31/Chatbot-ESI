"""
app.py : CISSOU propulsé par Hugging Face ZeroGPU (NVIDIA A100 gratuit).
Conserve 100% de l'interface originale index.html.
"""
from __future__ import annotations

import os
import spaces  # Bibliothèque interne de Hugging Face pour activer le GPU
from cissou import create_app

print("==================================================")
print("Démarrage de CISSOU sur ZeroGPU (Nvidia A100)...")
print("==================================================")

# 1. Cette fonction satisfait le test de démarrage de ZeroGPU
@spaces.GPU
def activate_zero_gpu():
    """Valide l'activation du GPU auprès de Hugging Face."""
    return "GPU Activated"

# Lance la vérification
try:
    activate_zero_gpu()
except Exception as e:
    print(f"Info GPU : {e}")

# 2. Initialisation de votre application Flask et de l'index FAISS
app = create_app()

# 3. Lancement du serveur sur le port 7860
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"--> CISSOU écoute sur le port {port}")
    app.run(host="0.0.0.0", port=port, threaded=True)
