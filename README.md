# skyhunt

Pipeline do wykrywania **wszystkich** obiektów ruchomych na wideo nieba 4K, także tych na granicy szumu. Każdy obiekt jest mierzony, a potem, jeśli się da, wyjaśniany: satelita (TLE, NORAD ID), meteor, samolot albo bliski obiekt (ptak, nietoperz, owad). Jako anomalię oznaczamy tylko to, co zostaje po odrzuceniu znanych klas, i to według kryteriów zamrożonych przed analizą.

Pełna specyfikacja: [docs/HANDOFF_skyhunt.md](docs/HANDOFF_skyhunt.md). Baseline CPU (tylko referencja wyników): [baseline/skytracks.py](baseline/skytracks.py).

## Status

| Kamień | Zakres | Stan |
|---|---|---|
| M0 | repo, config, notebook Colab, dekodowanie GPU z pomiarem, manifest i wznawianie | kod gotowy, czeka na pierwsze uruchomienie na Colab |
| M1 | tło, szum, maski, detektor per klatka, tory, regresja na pliku pilotażowym | – |
| M2 | plate solve, WCS, synchronizacja czasu po satelitach, identyfikacja TLE (NORAD) | – |
| M3 | shift-and-stack na GPU, FAR z tasowania, injection–recovery | – |
| M4 | tory nieliniowe, odległość z rozmycia, klasy biologiczne, meteory, samoloty | – |
| M5 | scoring anomalii, raport HTML, wycinki | – |

## Dane

| | |
|---|---|
| Aparat | Fujifilm X-E3, Fujinon XF 50mm F1.0; ostrość ręcznie, tuż przed ∞ |
| Wideo | 3840×2160, 24000/1001 fps, H.264 w MOV, GOP 24 klatki (1 s), **bez klatek B**, migawka 1/24 s |
| Miejsce | 51.718042 N, 19.582748 E, 210 m n.p.m. (Łódź) |
| `DSCF4641.MOV` | 320 s, start ok. 2026-09-27 18:26 UTC (zegar aparatu spieszył się o 10 min); najjaśniejsza gwiazda: Deneb |
| `dark_frames.MOV` | 320 s, zakryty obiektyw, 2026-09-28 (po korekcie zegara); do FAR i mapy hot pikseli |

Czas z metadanych jest tylko punktem startowym (±60 s). Ostateczną poprawkę zegara daje dopasowanie przelotów satelitów (M2). Data EXIF Fuji w `udta` to start nagrania; `mvhd.creation_time` wypada ~24 s po jego końcu (sprawdzone na obu plikach).

## Uruchomienie (Colab)

Otwórz [colab/run_skyhunt.ipynb](colab/run_skyhunt.ipynb) w Colab (GPU A100 lub L4) i uruchom komórki od góry. Nagrania trzymaj w `MyDrive/skyhunt/raw/`, wyniki trafiają do `MyDrive/skyhunt/out/<nazwa_pliku>/`.

CLI:

```bash
skyhunt probe RAW_DIR                      # metadane + wstępny czas (bez zapisu)
skyhunt bench-decode PLIK --out OUT/_bench # przepustowość backendów dekodowania
skyhunt run [RAW_DIR] [--out OUT]          # pipeline ze wznawianiem
skyhunt run PLIK --stages stack --force stack
skyhunt status [RAW_DIR]                   # stan etapów z manifestów
```

Każde polecenie przyjmuje `--config` oraz dowolną liczbę `--set klucz=wartość`, np. `--set decode.batch_frames=16`.

## Konfiguracja i wznawianie

- `config.yaml` zawiera wszystkie parametry. Sekcja `files` przechowuje nadpisania per plik, np. inną poprawkę zegara dla starszego nagrania albo `role: dark`.
- Każdy etap deklaruje, od których sekcji configu zależy. Jego hash liczy się z tych sekcji, z numeru `rev` etapu i z hashy etapów wymaganych.
- `manifest.json` per plik zapisuje dla każdego etapu: status, hash, wersję kodu (w tym commit git), czasy, pliki wynikowe i metryki. Etap jest pomijany, jeśli ma status `done` z tym samym hashem i jego pliki istnieją.
- Zmiana configu przelicza tylko zależne etapy. Zmiana logiki etapu wymaga podbicia `rev`.
- Wszystkie zapisy JSON są atomowe (plik tymczasowy + `os.replace`), więc przerwanie w trakcie nie psuje manifestu.

## Dekodowanie

Pracujemy na kanale Y (luminancja) w pełnej rozdzielczości. Backendy w kolejności prób:

| backend | gdzie dekoduje | Y | uwagi |
|---|---|---|---|
| `nvcodec` | NVDEC (PyNVVideoCodec) | dokładne | NV12 w pamięci GPU, bierzemy płaszczyznę Y |
| `torchcodec` | NVDEC | ±1 DN | zwraca RGB; tylko do benchmarku, bo `exact_luma_required: true` |
| `torchaudio` | NVDEC (`h264_cuvid`) | dokładne | `torchaudio.io` znika w nowszych wersjach torchaudio |
| `pyav` | CPU, wątek producenta | dokładne | transfer na GPU przez pinned memory |
| `ffmpeg` | CPU, podproces | dokładne | `extractplanes=y`, fallback bez zależności |

Parytet backendów sprawdzają testy (`tests/test_decode.py`): na prawdziwym nagraniu backendy „dokładne” muszą dać Y identyczne bit w bit z PyAV.

### Przepustowość

Do uzupełnienia po pierwszym uruchomieniu na Colab (`skyhunt bench-decode`, wynik w `out/_bench/decode_bench.md`).

## Wyniki etapów M0

| plik | opis |
|---|---|
| `meta.json` | metadane kontenera (GOP, klatki kluczowe, klatki B), prior czasu z kandydatami ze wszystkich źródeł |
| `stack_mean.npy/.png` | średnia z całego nagrania (do plate solve w M2) |
| `stack_max.npy`, `stack_max_minus_mean.png` | maksimum oraz max−średnia (podgląd torów) |
| `frame_stats.csv` | per klatka: czas, I/P, średnia, mediana, p99.9 |

Metryka `keyframe_pulse_dn` w manifeście to średnia różnica jasności I-klatek względem sąsiednich P-klatek. Wartość wyraźnie różna od zera oznacza pulsowanie co GOP, które trzeba maskować (sekcja 5.3 handoffu).

## Testy

```bash
python -m pytest -q
```

Testy z markerem `gpu` wymagają CUDA. Testy `video` wymagają zmiennej `SKYHUNT_TEST_VIDEO` wskazującej prawdziwe nagranie. Notebook ustawia ją automatycznie.

## Zasady interpretacji

- Kryteria „anomalii” są zamrożone w `config.yaml` przed przetworzeniem pełnego zbioru. Zmiana kryteriów wymaga nowej wersji i przeliczenia całości.
- Jedna kamera nie daje odległości ani prędkości liniowej. Wyjątkiem jest estymacja z rozmycia dla obiektów bliskich. Raport nigdy nie podaje km/s dla ostrych obiektów bez triangulacji.
- Czułość bez kontroli fałszywych alarmów jest bezwartościowa. Raport zawsze podaje limit detekcji (injection–recovery) i oczekiwaną liczbę fałszywych detekcji na godzinę nagrania.
- Rekomendacja sprzętowa:
  - druga kamera w odległości 5–20 m (paralaksa od razu rozstrzyga, czy obiekt jest bliski, czy daleki),
  - nagrania 1080p/60 fps obok 4K/24 fps (rozstrzyga aliasing trzepotu skrzydeł),
  - okresowo 1–2 min z zakrytym obiektywem, najlepiej w nocy, w temperaturze zbliżonej do sesji.
