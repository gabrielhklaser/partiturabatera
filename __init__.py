"""
DrumScribe — transcrição de bateria isolada em partitura.

Cadeia de processamento (tudo explícito, nada de caixa-preta):

    audio_io      decodificação, normalização, forma de onda
    dsp           espectrograma, fluxo por banda, detecção de onsets *por grupo*,
                  features físicas por evento (delta de banda, sub-razão, slopes…)
    grid          andamento (ACF + pente), refinamento por resíduo, grade métrica,
                  quantização com "grade mínima suficiente" e swing com histerese
    classify      escores e portões físicos por peça (bumbo, caixa, chimel, pratos, toms)
    rules         escrita rítmica: durações, pausas, vigas, acentos, ghost, flam, repetições
    layout        geometria da partitura (fonte única para SVG e PDF)
    engrave       PDF (reportlab) e SVG
    musicxml      MusicXML 4.0            midi_out   Standard MIDI File tipo 1
    pipeline      orquestra tudo em `transcribe_bytes` / `rebuild_score`
    server        aplicação web (upload → partitura → exportação)

Importe submódulos diretamente (`from drumscribe.pipeline import transcribe_file`).
"""

__version__ = "0.9.0"
__all__ = ["audio_io", "dsp", "grid", "classify", "rules", "layout", "engrave",
           "musicxml", "midi_out", "pipeline", "server"]
