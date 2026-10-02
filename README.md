# skyhunt

**Polski** | [English](README.en.md)

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

W nagraniach kolorowych PDF obiektu ma też stronę **kolor** (niżej), a `summary.pdf` stronę kalibracji koloru i kolumnę „kolor” w tabeli torów.

### Kolor

Etap `color` mierzy kolor każdego toru.
- **Kalibracja:** co minutę nagrania (3 epoki) mierzę kolor gwiazd z katalogu o znanym wskaźniku barwy B−V. Z nich wychodzi „linia gwiazd” na wykresie log(R/G) × log(B/G). To usuwa wpływ balansu bieli, symulacji filmu i łuny, a przy okazji wychodzi dryf balansu bieli w trakcie nagrania.
- **Obiekt:** w każdej klatce odejmuję tło z klatek, w których obiekt jest już dalej (gwiazdy i łuna znikają). Liczę tylko klatki nieprześwietlone.
- **Wynik:** temperatura barwowa `T_eq` (położenie na linii gwiazd), **nadmiar zieleni** (odległość od linii) i zmiana koloru w czasie. Pliki: `color_calib.json`, `track_color.csv`, `track_color_points.parquet`.
- **Kontrola:** zidentyfikowane satelity to światło Słońca odbite (B−V ≈ 0,6–0,9). Ich średni kolor jest odniesieniem dla innych torów.
- **Słaba kalibracja:** gdy linia gwiazd jest prawie płaska (nachylenie < `color.min_locus_slope`, oczekiwane ~0,3 dex na 1 mag B−V), kodek zgniótł kolor małych punktów. Wtedy raport nie podaje kelwinów, tylko „cieplejszy/chłodniejszy niż satelity o X dex” (`d_sun_dex`). Tak było w DSCF4651 (0,048 dex/mag). Temperatura poza skalą B−V −0,4…2,5 jest pokazywana jako granica, np. „T ≤ 2725 K (poza skalą)”.

Podpowiedzi (hipotezy, zawsze z liczbami):

| obiekt | kolor | podpowiedź |
|---|---|---|
| meteor | nadmiar zieleni | Mg 517 nm / O 557,7 nm? |
| meteor | ciepły (< 3500 K) | Na/Fe? |
| meteor | gorący (> 8000 K) | szybki, Ca/Mg? |
| inne | jak satelity | oświetlony Słońcem |
| inne | ciepły (< 3000 K) | łuna miasta (sód)? |
| inne | skaczący R/G | światła nawigacyjne? |

**Czego RGB nie powie:** kamera ma trzy szerokie pasma, więc składu meteoru (proporcji linii Na / Mg / Fe) nie da się z niej odczytać. Do tego potrzebna jest siatka dyfrakcyjna przed obiektywem (folia 500–1000 linii/mm). Obok meteoru pojawia się wtedy jego widmo.

**Ustawienia aparatu do koloru:** stały balans bieli (światło dzienne albo 5500 K, nie auto), symulacja Standard/Provia bez Color Chrome, **Kolor +4** (wzmacnia chromę, zanim H.264 ją zgniecie; kalibracja na gwiazdach to przelicza). Jasne obiekty są prześwietlone i nie mają koloru; PDF podaje, ile klatek odrzucono. Nagrania czarno-białe są wykrywane i pomijane.

### Sesje zdjęć RAW (w budowie)

Zamiast wideo można przetwarzać serie zdjęć RAW (Fujifilm RAF) z bracketingiem AE: **jeden podfolder w `raw/` = jedna sesja** (np. `raw/deneb_0210/`), wyniki w `out/<folder>/`. Dark: folder z „dark” w nazwie. Miejsce obserwacji i gwiazdę-podpowiedź wpisuje się w komórce „Nagrania” pod nazwą folderu.
- **Gotowe (F1):** EXIF i numer w serii (MakerNote Fuji), rytm interwałometru z pełnych sekund EXIF, plate solve, głęboki stos każdej klasy jasności (0, +1, −1 EV) wyrównany do obrotu nieba, raport z mapą i przebiegiem sesji.
- **W kolejnych etapach:** satelity jako kreski z dokładnym czasem i NORAD (F2), jasność i błyski wzdłuż kreski, kolor (F3), planetoidy na stosie (F4).
- **Ustawienia aparatu (X-E3):** M, f/1.0, ISO 800, migawka elektroniczna, tylko RAW (kompresja bezstratna), DR100, WB 5600 K, AE BKT ±1 EV (1/2 s, 1 s, 1/4 s), redukcja szumów długich czasów wyłączona, interwałometr.
- Diagnostyka plików przed pierwszą sesją: [colab/raw_diagnostics.ipynb](colab/raw_diagnostics.ipynb).

### Planetoidy, komety i NEO

Etap `smallbodies` sprawdza, które znane małe ciała były w kadrze, i mierzy je w nagraniu.
- **Lista:** JPL Small-Body Identification API (`sb_ident`), pozycje z numerycznego całkowania orbit, jasność V i ruch w ″/h. Osobne zapytanie o NEO, także słabsze. Do JPL idzie położenie obserwatora zaokrąglone do 0,1° (~10 km): dla NEO na 0,01 au zmienia to pozycję o ~1″, a dokładne miejsce zostaje prywatne.
- **Dlaczego nie tor:** planetoida pasa głównego przesuwa się przez 5 minut o ~0,1 px, więc w nagraniu wygląda jak gwiazda. Detektor torów wymaga ≥ ~0,1 px na klatkę, czyli dziesiątek tysięcy ″/h. Tor może dać tylko NEO przelatująca bardzo blisko Ziemi; wtedy raport podpisuje tor jej nazwą.
- **Pomiar:** stos wszystkich klatek w małym oknie, które jedzie razem z niebem i obiektem. Szum maleje jak √N, więc zasięg jest o kilka magnitudo głębszy niż w jednej klatce. Punkt zerowy jasności z gwiazd katalogowych w tym samym stosie, z poprawką na winietowanie.
- **Werdykt:** „wykryta” = SNR ≥ 5 w przewidzianym miejscu, jasność zgodna z przewidywaną (±1 mag) i brak jaśniejszej gwiazdy tła (Gaia DR3 z VizieR) w aperturze. Inaczej „zlewa się z gwiazdą”, „za słaba (zasięg X mag)” albo „niewykryta”.
- **Raport:** różowe romby na mapie (wypełniony = wykryta), tabela z jasnością przewidzianą i zmierzoną, zasięgiem, ruchem i odchyłką O−C, miniatury stosów. Pliki: `smallbodies.csv`, `smallbodies.json`, `smallbodies_tracks.csv`.

### `tracks_final.csv`: tabela wszystkich torów

Plik jest w `out/<plik>/tracks_final.csv`, jeden wiersz na tor. Obok leży ta sama tabela jako `.parquet`. To najwygodniejsze miejsce, żeby znaleźć konkretny obiekt, zanim otworzysz PDF-y.

**Otwieranie.**
- Colab: `pandas.read_csv`, przykłady niżej.
- Arkusze Google / Excel: liczby są zapisane z kropką dziesiętną. Przy polskich ustawieniach regionalnych ustaw najpierw region arkusza na „Stany Zjednoczone” (Arkusze: Plik → Ustawienia) albo importuj z separatorem `,` i ustawieniami angielskimi. Inaczej `0.771` zamieni się w datę albo tekst.

**Najważniejsze kolumny.**

| kolumna | znaczenie |
|---|---|
| `track_id` | numer toru, ten sam w nazwach plików: `objects/sat_<NORAD>_t<id>.pdf`, `objects/unid_t<id>.pdf`, `clips/t<id>.mp4` |
| `kind` | `sat` (zidentyfikowany satelita) albo `unid` (niezidentyfikowany) |
| `norad`, `sat_name`, `confidence`, `match_reason` | identyfikacja: numer NORAD, nazwa, pewność (`high`/`medium`/`low`), uzasadnienie |
| `tau0`, `tau1`, `dur_s` | początek i koniec w sekundach od startu filmu (≈ licznik odtwarzacza) oraz czas trwania |
| `utc_start`, `utc_end` | początek i koniec w UTC, już po poprawce zegara z satelitów |
| `omega_deg_s` | prędkość kątowa [°/s]; LEO nad głową to ~0,5–1,1 °/s |
| `curv_arcsec` | odchylenie toru od koła wielkiego [″]; satelity zwykle < 20″ |
| `ra0`, `dec0`, `ra1`, `dec1` / `az0`, `alt0`, `az1`, `alt1` | położenie na niebie na początku i końcu toru (RA/Dec ICRS oraz azymut i wysokość) |
| `n` | liczba punktów toru (klatek z detekcją) |
| `peak_snr_median` | jasność jako SNR; próg wykrycia to 5, wyraźne obiekty > 15, prześwietlone > 50 |
| `speed_px_frame`, `x0`, `y0`, `x1`, `y1` | ruch i położenie w pikselach (kadr 3840×2160) |
| `cross_ratio` | szerokość obiektu ÷ szerokość gwiazdy; > 1,8 znaczy nieostry (bliski), chyba że obiekt jest bardzo jasny |
| `f_peak_hz`, `f_alias_hz`, `f_power` | modulacja jasności (błyski, obrót); przy 24 kl/s nie da się odróżnić `f_peak_hz` od `f_alias_hz` |
| `starts_inside`, `ends_inside` | `False` oznacza, że obiekt wlatuje lub wylatuje przez krawędź kadru; `True` na końcu toru oznacza, że gaśnie w kadrze (np. wejście w cień Ziemi) |
| `class_hint`, `class_reason` | podpowiedź klasy dla niezidentyfikowanych: `satelita?`, `meteor?`, `samolot?`, `bliski obiekt?` |
| `along_sigma_px`, `streak_ratio` | kształt śladu w klatce: wydłużenie wzdłuż ruchu (kreska meteoru) i stosunek do szerokości w poprzek |
| `flock_n` | liczba równoległych torów o podobnej prędkości (przelot ptaków); 0 = brak grupy |

Kolor torów jest w osobnym pliku `track_color.csv` (ten sam `track_id`): `T_eq_K`, `bv_eq`, `e_bv_eq`, `green_excess`, `d_sun_dex` (kolor względem satelitów, „+” = cieplejszy), `color_hint`, `n_color`, `n_saturated`.

**Przykłady (komórka w Colab).**

```python
import pandas as pd
t = pd.read_csv(f'{OUT}/DSCF4641/tracks_final.csv')

# zidentyfikowane satelity w kolejności pojawienia się
t[t.kind == 'sat'].sort_values('tau0')[['track_id', 'tau0', 'utc_start', 'norad', 'sat_name', 'omega_deg_s']]

# co było w kadrze w 43. sekundzie filmu
T = 43
t[(t.tau0 <= T + 2) & (t.tau1 >= T - 2)]

# niezidentyfikowane warte obejrzenia: dłuższe tory, najjaśniejsze na górze
t[(t.kind == 'unid') & (t.n >= 15)].sort_values('peak_snr_median', ascending=False)
```

**Szum.** Tor z `n` ≤ 8, `peak_snr_median` ≈ 5–6 i skokami 25–48 px na klatkę to prawie na pewno przypadkowo połączone detekcje szumu, a nie obiekt. W `DSCF4641` to ~250 z 323 niezidentyfikowanych torów.

### Weryfikacja i zgłaszanie obserwacji

**Sprawdzenie identyfikacji.** W PDF `sat_…` zielony tor (pomiar) powinien pokrywać się z przerywaną predykcją z elementów orbit. W tabeli obok są residuum poprzeczne (kilkadziesiąt ″), δ_j − Δ (< ~1 s), zgodność prędkości i kierunku oraz oświetlenie przez Słońce. Satelita „w cieniu Ziemi” byłby niewidoczny, więc takie dopasowanie jest podejrzane. Niezależnie można to sprawdzić w Stellarium (wtyczka Satellites): wystarczy ustawić miejsce i czas `utc_start` z `tracks_final.csv`.

**Satelity spoza publicznych katalogów.** Poza CelesTrak i Space-Track pipeline pobiera katalog `classfd` (Mike McCants, <https://mmccants.org/tles/>). To elementy satelitów, głównie wojskowych, śledzonych przez amatorską sieć obserwatorów. Elementy bywają sprzed wielu dni, dlatego dopasowanie do nich jest tylko kandydatem z pewnością `low`, widocznym w PDF niezidentyfikowanego obiektu.

**Zgłaszanie.** Space-Track nie przyjmuje obserwacji od amatorów. Pozycje satelitów zgłasza się społeczności obserwatorów (lista SeeSat-L, <https://www.satobs.org>) w formacie IOD. Pipeline zapisuje je w `report/iod.txt`:
- po 3 pozycje (początek, środek, koniec toru) dla zidentyfikowanych satelitów oraz dla niezidentyfikowanych torów, które są długie, prawie proste i wyraźnie nad szumem (sekcja `iod` w configu);
- RA/Dec J2000 w formacie 1, niepewność czasu z synchronizacji, niepewność pozycji ze zgodności epok plate solve.

Przed wysłaniem:
- poproś na SeeSat-L o numer stacji i wpisz go w `iod.station`, bo 9999 oznacza nieprzydzielony;
- zaznacz w zgłoszeniu, że czas jest kalibrowany na satelitach z katalogu, a nie z GPS.

**Synchronizacja czasu.** Pozycje torów na niebie liczymy z plate solve i modelu nieruchomej kamery: piksel ↔ stały kierunek Alt/Az. Nie zależą one od błędu zegara. Tory proste o prędkościach LEO porównujemy z przelotami z elementów orbit (SGP4) w oknie ±5σ wokół czasu z metadanych, a gdy to nie wystarczy, w ±2 h.

- Poprawka Δ to **mediana δ zgodnych satelitów**: każdy satelita liczy się raz, a elementy amatorskie (classfd) są pomijane, gdy publicznych jest co najmniej 3. Raport nadal pokazuje pierwszego zidentyfikowanego satelitę jako odniesienie.
- Wcześniej Δ brałem z pierwszego satelity. Dwie części jednego nagrania (DSCF4647/4648, ten sam zegar aparatu) różniły się wtedy o ≥ 0,3 s, bo błąd elementów orbity jednego Starlinka przechodził na cały czas. Stary tryb: `identify.reference: first`.
- Uznajemy Δ, gdy co najmniej 2 niezależne tory dają zgodne δ (±1 s). Pojedyncze dopasowanie ma pewność `low`, bo równoległe powłoki Starlinka łatwo pomylić.
- Pozostałe satelity identyfikujemy już przy ustalonym Δ i raportujemy rozrzut ich δ.

**Elementy orbit.** CelesTrak udostępnia tylko bieżące elementy, więc snapshot trzeba zamrozić krótko po nagraniu (komórka w notebooku). Jest zapisywany w `cache/gp/` na Drive, a kopia użytych elementów trafia do `gp_elements.csv` przy wynikach. Format to CSV/OMM, bo numery NORAD ≥ 100000 nie mieszczą się w TLE. Space-Track (historia elementów, pełny katalog z członami rakiet i śmieciami) jest opcjonalny: login wpisujesz w Colab Secrets.

Dane konstelacji i nazw gwiazd: [d3-celestial](https://github.com/ofrohn/d3-celestial) (BSD-3, © Olaf Frohn), w `skyhunt/data/d3celestial/`.

## Dane

| | |
|---|---|
| Aparat | Fujifilm X-E3, Fujinon XF 50mm F1.0; ostrość ręcznie, tuż przed ∞ |
| Wideo | 3840×2160, H.264 w MOV, **bez klatek B**. DSCF4641: 24000/1001 fps, migawka 1/24 s, B&W. Od 28.09.2026: 30000/1001 fps, migawka 1/30 s, ISO auto, kolor (symulacja Standard). fps i czas klatki są czytane z pliku |
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
