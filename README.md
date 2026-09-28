# skyhunt

## Od autora

Ten kod piszę dla wszystkich obserwatorów nieba, nie tylko dla siebie. Nagrywam niebo zwykłym aparatem z jasnym obiektywem i chcę wiedzieć, co naprawdę przeleciało przez kadr: który to satelita (z numerem NORAD), co jest meteorem, co samolotem, a co zostaje niewyjaśnione. Wszystko ma być policzone i sprawdzalne, bez zgadywania. Jeśli masz aparat, statyw i trochę cierpliwości, możesz robić to samo.

Kod jest otwarty (licencja MIT): używaj go, zmieniaj, rozwijaj. **Mam jedną prośbę: jeśli korzystasz z tego projektu w swoich obserwacjach, publikacjach, filmach czy we własnym kodzie, wspomnij o nim** i podaj link do repozytorium: <https://github.com/LukaszCiesielski-lodz/skytrack_uap>. Chętnie też zobaczę Twoje wyniki. Zgłoszenia błędów, pomysły i poprawki (issues, pull requesty) są mile widziane.

## Co to robi

Pipeline do wykrywania **wszystkich** obiektów ruchomych na wideo nieba 4K, także tych na granicy szumu. Każdy obiekt jest mierzony, a potem, jeśli się da, wyjaśniany: satelita (TLE, NORAD ID), meteor, samolot albo bliski obiekt (ptak, nietoperz, owad). Jako anomalię oznaczamy tylko to, co zostaje po odrzuceniu znanych klas, i to według kryteriów zamrożonych przed analizą.

Pełna specyfikacja: [docs/HANDOFF_skyhunt.md](docs/HANDOFF_skyhunt.md). Baseline CPU (tylko referencja wyników): [baseline/skytracks.py](baseline/skytracks.py).

## Status

| Kamień | Zakres | Stan |
|---|---|---|
| M0 | repo, config, notebook Colab, dekodowanie GPU z pomiarem, manifest i wznawianie | działa na Colab (L4) |
| Raport | detektor per klatka, tory, plate solve, synchronizacja czasu po satelitach, NORAD, PDF z konstelacjami i wycinkami | działa; pierwsze wyniki niżej |
| M3 | shift-and-stack na GPU (słabe obiekty), FAR z tasowania, injection–recovery | – |
| M4 | tory nieliniowe, odległość z rozmycia, klasy biologiczne, meteory, samoloty (ADS-B) | – |
| M5 | scoring anomalii | – |

### Pierwsze wyniki: `DSCF4641.MOV` (27.09.2026, Łabędź w zenicie, 320 s)

| | |
|---|---|
| Plate solve | 5/5 epok; pole 25,8° × 14,7°, 24,7″/px, bez cropu 4K; zgodność epok 0,44 px |
| Detekcja | 4,04 mln detekcji, 359 torów; NVDEC ~150 kl/s (stack), ~50–60 kl/s (detekcja) |
| Poprawka zegara | Δ = +26,85 ± 0,14 s, zgodna dla 24 torów satelitów (zegar aparatu spieszył się o 9 min 33 s) |
| Katalog | 31 354 obiekty (CelesTrak + Space-Track) |
| Zidentyfikowane | 36 torów, m.in. Starlink, Kuiper, Hulianwang, Globalstar; najjaśniejszy obiekt nagrania to STARLINK-2112 (NORAD 47391) |
| Położenie z paralaksy satelitów | 0,24 km od wpisanych współrzędnych |
| Nagranie ciemne | ~112 fałszywych torów na godzinę, 6 gorących pikseli |

Lekcja z tego nagrania: pierwszy przebieg dał 0 identyfikacji, bo w configu było miejsce oddalone o ~2,2 km od faktycznego. Satelity na ~500 km były przez to przesunięte o ~0,25° (paralaksa). Dlatego współrzędne wpisuje się teraz osobno dla każdego nagrania, a pipeline sam sprawdza je z paralaksy.

## Raport

Dla każdego nagrania nieba w `out/<plik>/report/`:

| plik | zawartość |
|---|---|
| `summary.pdf` | cały kadr z konstelacjami i wszystkimi torami; poprawka czasu i jej zgodność między satelitami; tabela torów; satelity przewidziane w kadrze, a niewykryte; FAR z nagrania ciemnego |
| `objects/sat_<NORAD>_t<id>.pdf` | zidentyfikowany satelita: tło gwiazd, konstelacje (np. Łabędź), nazwy gwiazd, tor na zielono z kreskami co 1 s, predykcja z elementów orbit; NORAD, nazwa, COSPAR, odległość, wysokość, oświetlenie; strona 2: pasek 12 klatek z ≥ 1 s nagrania; klip MP4 w załączniku PDF |
| `objects/unid_t<id>.pdf` | obiekt niezidentyfikowany: tor **na czerwono** na tle gwiazd; prędkość kątowa [°/s], czas przelotu, początek i koniec w UTC (± niepewność poprawki czasu), RA/Dec i Az/Alt, podpowiedź klasy, niesprawdzone hipotezy |
| `clips/t<id>.mp4` | wycinek ≥ 1 s wokół toru, obiekt zaznaczony okręgiem |

**Synchronizacja czasu.** Pozycje torów na niebie liczymy z plate solve i modelu nieruchomej kamery: piksel ↔ stały kierunek Alt/Az. Nie zależą one od błędu zegara. Tory proste o prędkościach LEO porównujemy z przelotami z elementów orbit (SGP4) w oknie ±5σ wokół czasu z metadanych, a gdy to nie wystarczy, w ±2 h.

- Poprawka Δ pochodzi z **pierwszego zidentyfikowanego satelity**.
- Uznajemy ją, gdy co najmniej 2 niezależne tory dają zgodne δ (±1 s). Pojedyncze dopasowanie ma pewność `low`, bo równoległe powłoki Starlinka łatwo pomylić.
- Pozostałe satelity identyfikujemy już przy ustalonym Δ i raportujemy rozrzut ich δ.

**Elementy orbit.** CelesTrak udostępnia tylko bieżące elementy, więc snapshot trzeba zamrozić krótko po nagraniu (komórka w notebooku). Jest zapisywany w `cache/gp/` na Drive, a kopia użytych elementów trafia do `gp_elements.csv` przy wynikach. Format to CSV/OMM, bo numery NORAD ≥ 100000 nie mieszczą się w TLE. Space-Track (historia elementów, pełny katalog z członami rakiet i śmieciami) jest opcjonalny: login wpisujesz w Colab Secrets.

Dane konstelacji i nazw gwiazd: [d3-celestial](https://github.com/ofrohn/d3-celestial) (BSD-3, © Olaf Frohn), w `skyhunt/data/d3celestial/`.

## Dane

| | |
|---|---|
| Aparat | Fujifilm X-E3, Fujinon XF 50mm F1.0; ostrość ręcznie, tuż przed ∞ |
| Wideo | 3840×2160, 24000/1001 fps, H.264 w MOV, GOP 24 klatki (1 s), **bez klatek B**, migawka 1/24 s |
| Miejsce | **osobno dla każdego nagrania**: komórka „Miejsce obserwacji” w notebooku zapisuje je do `MyDrive/skyhunt/sites.yaml` (poza repo). Sekcja `site` w `config.yaml` to tylko wartość domyślna (Łódź). Błąd ~2 km psuje identyfikację satelitów; pipeline ostrzega, gdy paralaksa wskazuje przesunięcie > 0,3 km |
| `DSCF4641.MOV` | 320 s, start 2026-09-27 18:26:35 UTC (po synchronizacji; zegar aparatu spieszył się o 9 min 33 s); najjaśniejsza gwiazda: Deneb |
| `dark_frames.MOV` | 320 s, zakryty obiektyw, 2026-09-28 (po korekcie zegara); do FAR i mapy hot pikseli |

Czas z metadanych jest tylko punktem startowym (±60 s). Ostateczną poprawkę zegara daje dopasowanie przelotów satelitów (M2). Data EXIF Fuji w `udta` to start nagrania; `mvhd.creation_time` wypada ~24 s po jego końcu (sprawdzone na obu plikach).

## Uruchomienie (Colab)

Otwórz [colab/run_skyhunt.ipynb](colab/run_skyhunt.ipynb) w Colab (GPU A100 lub L4) i uruchom komórki od góry. Nagrania trzymaj w `MyDrive/skyhunt/raw/`, wyniki trafiają do `MyDrive/skyhunt/out/<nazwa_pliku>/`.

Przy każdej nowej obserwacji:
1. Wpisz jej współrzędne w komórce „Miejsce obserwacji”.
2. Krótko po nagraniu uruchom komórkę „Snapshot elementów orbit” (CelesTrak ma tylko bieżące elementy).
3. Opcjonalnie dodaj login Space-Track w Colab Secrets (`SPACETRACK_USER`, `SPACETRACK_PASSWORD`).

Nie zapisuj notebooka z wpisanymi współrzędnymi z powrotem do publicznego repo.

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

Colab, GPU L4, 4K H.264: `nvcodec` ok. 150–165 kl/s w etapie `stack`; detekcja z tłem, filtrem i etykietowaniem na GPU ok. 50–65 kl/s (320 s nagrania w ~2,5 min). Szczegóły: `skyhunt bench-decode`, wynik w `out/_bench/decode_bench.md`.

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
