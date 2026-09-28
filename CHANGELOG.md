# CHANGELOG

## Raport PDF, synchronizacja czasu, NORAD — 2026-09-28 (0.2.0)

**Dodane (kod i testy offline; nic jeszcze nie uruchomione na prawdziwych danych):**
- **`detect`** (GPU, strumieniowo):
  - tło = mediana ±1 s co 24 klatki, szum = MAD z podłogą, offset per klatka;
  - matched filter, histereza SNR 5/3, komponenty (cupy lub scipy), momenty na GPU;
  - stacki epok co 60 s do plate solve; szerokość PSF gwiazd.
- **`tracks`**:
  - filtr statyczny z warunkiem czasu (wolne obiekty zostają);
  - łączenie z przypisaniem globalnym, sklejanie fragmentów;
  - pomiary: prędkość, krzywizna, szerokość ÷ gwiazda, widmo jasności z aliasem, pojawienie/zniknięcie w kadrze.
- **`astrometry`**:
  - `solve-field` z podpowiedzią (Deneb), SIP 3. rzędu, indeksy 4110–4119 z MD5;
  - zgodność epok w stałym układzie Alt/Az;
  - rzeczywiste pole widzenia i werdykt cropu 4K.
- **`tle`**: snapshot CelesTrak (CSV/OMM, bez ponawiania przy 403), opcjonalnie Space-Track `gp_history`, zamrożony `gp_elements.csv`.
- **`identify`**:
  - model nieruchomej kamery, więc pozycje torów na niebie nie zależą od błędu zegara;
  - propagacja SGP4 całego katalogu i przesiew;
  - Δ z pierwszego zidentyfikowanego satelity, potwierdzone ≥ 2 torami;
  - identyfikacja NORAD z pewnością i uzasadnieniem, oświetlenie (de421), satelity przewidziane w kadrze.
- **`report`**:
  - `summary.pdf`, PDF per obiekt (satelita na zielono z NORAD, niezidentyfikowany na czerwono z °/s, czasem przelotu i UTC);
  - konstelacje i nazwy gwiazd, pasek klatek ≥ 1 s, klip MP4 obok i w załączniku PDF.
- **`darkstats`** (nagranie z zakrytym obiektywem): gorące piksele, fałszywe tory na godzinę.
- **Role plików** w pipeline: `sky` / `dark`.
- **Notebook**: komórka zamrażająca snapshot CelesTrak, astrometry.net, opcjonalny Space-Track z Colab Secrets.

**Zmienione:** pilot `Trim2` i test regresji na nim pominięte (decyzja użytkownika).

**Uwaga:** CelesTrak był nieosiągalny z komputera lokalnego (blokada sieci), więc snapshot pobiera komórka w notebooku na Colab.

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
