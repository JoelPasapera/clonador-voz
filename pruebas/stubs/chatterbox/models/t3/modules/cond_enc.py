"""T3Cond simulado: mismos campos que chatterbox 0.1.7 (test_fidelidad.py lo comprueba)."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class T3Cond:
    speaker_emb: torch.Tensor
    clap_emb: Optional[torch.Tensor] = None
    cond_prompt_speech_tokens: Optional[torch.Tensor] = None
    cond_prompt_speech_emb: Optional[torch.Tensor] = None
    emotion_adv: Optional[torch.Tensor] = 0.5

    def to(self, *, device=None, dtype=None):
        for clave, valor in self.__dict__.items():
            if torch.is_tensor(valor):
                setattr(self, clave, valor.to(device))
        return self
