# CHANGELOG

## M0 — 2026-09-28 (0.1.0)

**Działa (kod, testy jednostkowe):**
- Pakiet `skyhunt/` z CLI `skyhunt probe | bench-decode | run | status` i `config.yaml`.
  - Nadpisania per plik w configu: zegar DSCF4641 spieszył się o 600 s; `dark_frames.MOV` ma `role: dark`; Deneb jako podpowiedź dla plate solve.
- Parser atomów MP4/MOV bez ffprobe, odczyt w milisekundach.
  - Wyciąga rozdzielczość, fps, liczbę klatek, pozycje klatek kluczowych, obecność klatek B, `mvhd.creation_time` i datę EXIF Fuji.
- Wstępny czas startu z metadanych z poprawką zegara i strefy (prior do synchronizacji po satelitach w M2).
- Pięć backendów dekodowania Y z automatycznym fallbackiem: `nvcodec`, `torchcodec`, `torchaudio`, `pyav`, `ffmpeg`.
  - `bench-decode` mierzy przepustowość i parytet Y między backendami.
- Manifest per plik i wznawianie po rozłączeniu.
  - Hash etapu liczony z sekcji configu, `rev` etapu i etapów wymaganych.
  - Zapisy atomowe; plik wideo rozpoznawany po nazwie, rozmiarze i hashu nagłówka.
- Etap `stack`: średnia, max, max−średnia, statystyki per klatka, metryka pulsowania I-klatek.
- Notebook `colab/run_skyhunt.ipynb`: montuje Drive, klonuje repo, instaluje zależności, uruchamia testy, benchmark i pipeline.

**Ustalone z metadanych:**
- Oba pliki: 3840×2160 H.264, 24000/1001 fps, 7680 klatek (320,3 s).
- GOP 24 klatki; brak klatek B (tylko I/P).
- Data Fuji w `udta` to start nagrania; `mvhd` wypada ~24 s po końcu.
- `DSCF4641.MOV`: start ≈ 2026-09-27 18:26:08 UTC (±60 s).

**Liczby:**
- Przepustowość dekodowania: brak (czeka na pierwsze uruchomienie na Colab).
- Limit mag i FAR: nie dotyczy M0.

**Nie sprawdzone:** kodu nie uruchamiano jeszcze nigdzie; lokalnie nie ma Pythona. Pierwszym testem jest komórka `pytest` w notebooku.
