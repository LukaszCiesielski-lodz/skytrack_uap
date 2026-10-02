# CHANGELOG

## Sesje zdjęć RAW — F1: wejście, czas, astrometria, głęboki stos — 2026-10-02

- **Nowe wejście:** podfolder w `raw/` z ≥ 3 plikami RAF = jedna sesja zdjęć (osobny rejestr etapów `PHOTO_PIPELINE`, wyniki w `out/<folder>/`). Etapy wideo i ich hashe bez zmian.
- `probe`: EXIF z JPEG-a w RAF (Pillow, bez exiftool) i MakerNote Fuji (numer w serii, licznik); serie bracketingu, klasy `ev0`/`ev+1`/`ev-1`; rytm interwałometru z pełnych sekund EXIF (noniusz: T_k = T0 + P·k); chwile otwarcia migawki; czas a priori w UTC → `photos.csv`, `meta.json`.
- `frames` → `astrometry`: zdjęcia ev0 co minutę jako FITS z superpikseli 3×3 (X-Trans: 5 G + 2 R + 2 B), plate solve wspólnym rdzeniem z wideo (`solve_epochs`).
- `process`: wszystkie zdjęcia (kopiowanie z Drive w wątkach), cache luminancji na dysku lokalnym, statystyki zdjęć (tło, szum, dryf, nasycenie), stosy per klasa wyrównane modelem nieruchomej kamery z odrzucaniem kresek, stos RGB ev0; punkty kontrolne co 60 zdjęć.
- `report`: mapa nieba na stosie z konstelacjami, przebieg sesji, podgląd koloru.
- Config: `photo`, `photo_overrides` (astrometria bez zmniejszania, `color.transfer: linear`), `input.photo_extensions`. Zależność `rawpy` (extra `raw`).
- Ustalenia z diagnostyki F0 (X-E3): czas EXIF co 1 s bez ułamków, każde zdjęcie ma własny czas, numer w serii 1-2-3, 14 bit przy migawce elektronicznej, czerń 1019, biel 16383.

## Poprawki po sesji 1.10.2026 — 2026-10-02

- Raport i IOD nie wywracają się na nagraniu bez torów (pusta `tracks_final` bez kolumn); `smallbodies` też (`KeyError: 'kind'`).
- Synchronizacja: bez kandydatów na satelity nie ma szerokiego przesiewu ±2 h (DSCF4663/4665: 15–28 min liczenia na pustym nagraniu).
- `smallbodies`: błąd fotometrii = większy z wzoru i z rozrzutu apertur wokół obiektu. Kompresja H.264 wygładza szum pikseli, więc sam wzór dawał zasięg stosu 17–30 mag. JPL: limit 180 s i 3 próby.

## Gwiazda-podpowiedź dla każdego pliku — 2026-10-01

- Komórka „Nagrania”: opcjonalne 4. pole w `NEW_SITES` (np. `(lat, lon, wys, 'Altair')`) trafia do `sites.yaml` jako `hint_star` i nadpisuje `astrometry.hint_star` tylko dla tego pliku. Plate solve kadru bez Denebu (Kasjopeja, Orzeł) nie traci wtedy czasu na próbę z błędną podpowiedzią.

## Planetoidy, komety i NEO — 2026-10-01

- **Etap `smallbodies`** (po `identify`, przed `report`; sekcja `smallbodies` w configu):
  - lista małych ciał w kadrze z JPL `sb_ident` (V ≤ 13, NEO ≤ 16), położenie obserwatora wysyłane w zaokrągleniu do 0,1°;
  - jedno przejście dekodera: stos okien jadących z niebem i obiektem (do 15 obiektów) oraz okien gwiazd katalogowych 4–6 mag do punktu zerowego `ZP = a + b·r²` (winietowanie);
  - fotometria, centroid (O−C), zasięg 5σ, sprawdzenie gwiazd tła w Gaia DR3 (VizieR), werdykt;
  - bardzo bliskie NEO dopasowywane do niezidentyfikowanych torów (pozycja ≤ 0,2°, tempo ×0,5…2).
- **Raport:** romby na mapie, tabela i miniatury stosów w `summary.pdf`, notka z zasięgiem; brak sieci lub błąd JPL nie blokuje raportu. `report` rev 5.

## Przelot ptaków, słaba kalibracja koloru — 2026-09-30

Po DSCF4651 (29.09.2026):
- **Przelot ptaków** (`classify.flock_*`): ≥ 3 niezidentyfikowane tory o podobnej prędkości (±25%) i kierunku (±20°), na różnych liniach, szybsze niż LEO (≥ 1,5°/s). „Bliski obiekt?” / „niesklasyfikowany” → „ptak?” z uzasadnieniem „przelot: N torów równolegle”. „Meteor?” zostaje meteorem, dostaje tylko drugą hipotezę. Nowa kolumna `flock_n`. W DSCF4651 7 torów 1,8–2,05°/s, wszystkie ze wschodu na zachód, ciepłe: nocna migracja oświetlona łuną; trzy z nich dostały wcześniej „meteor?”. `identify` rev 3.
- **Słaba kalibracja koloru** (`color.min_locus_slope: 0.1`): linia gwiazd w DSCF4651 miała nachylenie 0,048 dex/mag B−V (oczekiwane ~0,3), więc B−V gwiazdy było niepewne o ±1,2 mag. Wtedy podpowiedzi porównują kolor z satelitami (`d_sun_dex`, „cieplejszy/chłodniejszy niż satelity o X dex”) zamiast kelwinów, a próg nadmiaru zieleni rośnie do RMS linii. Ostrzeżenie w logu i na stronie kalibracji.
- **T_eq:** niepewność `e_bv_eq` (pomiar ⊕ rozrzut gwiazd / nachylenie); poza skalą B−V −0,4…2,5 raport pisze „T ≤ 2725 K (poza skalą)” zamiast udawanego pomiaru.
- **Apertura koloru** 4 → 6 px, pierścień [9, 14], izolacja 18 px: chroma 4:2:0 po H.264 jest rozmyta szerzej niż jasność. `color` rev 3.
- Raport: strona ograniczeń opisuje poprawkę zegara z mediany (było: „z pierwszego satelity”); krótsza podpowiedź w kolumnie „kolor”. `report` rev 4.

## Poprawka zegara z mediany satelitów — 2026-09-29

- `identify.reference: median` (decyzja użytkownika): Δ = mediana δ zgodnych satelitów, każdy NORAD raz, bez classfd przy ≥ 3 publicznych; σ = 1,2533·σ_MAD/√n (obejmuje błędy elementów wzdłuż orbity).
- Powód: DSCF4647 i DSCF4648 to jedno nagranie podzielone przez aparat, a z `first` wyszło Δ +4,95 i +3,64 s — niezgodność ≥ 0,3 s mimo tego samego zegara.
- Raport nadal pokazuje pierwszego zidentyfikowanego satelitę jako odniesienie; metoda w `time_sync.json` ma dopisek „Δ = mediana N satelitów”.

## Kolor torów — 2026-09-29

- **Etap `color`** (po `identify`, przed `report`; sekcja `color` w configu).
  - Odczyt RGB tylko dla klatek z torami, w małych wycinkach kadru: ffmpeg `rgb48le` (opcjonalnie NVDEC), a bez niego PyAV.
  - Kalibracja na gwiazdach z katalogu (B−V z d3-celestial, 3 epoki): „linia gwiazd” log R/G, log B/G ~ B−V, dopasowanie odporne (Theil–Sen + odrzucanie).
  - Kontrole: dryf balansu bieli między epokami, liniowość (log G vs mag ≈ −0,4), kolor satelitów (Słońce odbite).
  - Kolor obiektu: odjęcie tła z klatek f±k, apertura wydłużona wzdłuż ruchu (kreski meteorów), bez klatek prześwietlonych.
  - Wynik: `T_eq`, `bv_eq`, nadmiar zieleni, zmiana koloru w czasie, podpowiedź (meteor: Mg/O, Na/Fe, Ca/Mg; inne: Słońce, łuna miasta, światła nawigacyjne).
  - Nagrania czarno-białe są wykrywane i pomijane.
- **Raport:** strona „kolor” w PDF obiektu (kolor w czasie, wykres barw z gwiazdami i satelitami, kolorowe miniatury), strona kalibracji i kolumna „kolor” w `summary.pdf`. `report` rev 3.
- `sky.stars()` zwraca też `bv`.
- Po pierwszym przebiegu (DSCF4647/4648): kalibracja na 191/184 gwiazdach, RMS 0,053/0,069 dex, liniowość −0,40/−0,39, dryf balansu bieli 0,02/0,002 dex. Satelity B−V 1,03/0,82 (czerwieńsze od Słońca), więc zakres kontrolny rozszerzony do 0,5–1,2.
  - „Światła nawigacyjne?” tylko przy istotnym miganiu jasności w paśmie samolotów i ≥ 20 klatkach z kolorem. Wcześniej dostawały je ciepłe, wolne obiekty bez migania (łuna miasta). `color` rev 2.

## Mniej fałszywych torów, sklejanie pociętych — 2026-09-29

- **Odrzucanie łańcuchów szumu** (`tracks.weak_min_len: 10`, `tracks.weak_snr: 7`): tor krótszy niż 10 punktów i zarazem z medianą SNR maksimum < 7 jest odrzucany.
  - DSCF4647: tak wyglądało 301 z 334 torów bez dopasowania; DSCF4648: 260 z 286.
  - Wśród 189 zidentyfikowanych satelitów obu części nie było żadnego takiego toru (najsłabszy: 10 punktów, SNR 6,4).
  - Krótkie jasne tory (meteory, błyski) i długie słabe zostają.
  - Zostają też krótkie słabe tory z detekcjami wydłużonymi wzdłuż ruchu, zgodnie z prędkością (kreska meteoru w ekspozycji 1/30 s; `streak_min_along_frac`, `streak_min_ratio`). Kropki szumu są okrągłe.
  - Nowe kolumny `along_sigma_px` i `streak_ratio` w `tracks_final`.
  - Ograniczenie, które zostaje: meteory szybsze niż ~10°/s (> 48 px/klatkę, `init_gate_px`) nie tworzą toru wcale — do osobnego detektora kresek.
- **Sklejanie fragmentów** do 60 klatek przerwy (było 24). Jasny obiekt #9 w DSCF4647 był pocięty na 4 tory (#9, #19, #23, #27) przerwami 33–48 klatek.
- **Raport** usuwa stare PDF-y i klipy przed wygenerowaniem nowych (numery torów się zmieniają).
- `tracks` rev 3: przeliczy się `tracks`, `identify`, `adsb`, `report` (bez ponownej detekcji).

## Samoloty z ADS-B, ruch toru, „ptak?” — 2026-09-29

- **Etap `adsb`:** niezidentyfikowane tory są porównywane z trasami samolotów z historii ADS-B.
  - Źródło: adsb.lol, licencja ODbL. OpenSky REST daje tylko ostatnią godzinę, więc do starszych nagrań się nie nadaje.
  - Archiwum dnia (kilka GB, podzielony tar) jest czytane strumieniowo. W `cache/adsb/` na Drive zostają tylko punkty w promieniu 150 km.
  - Dopasowanie: mediana odległości kątowej tor ↔ trasa ≤ 1,5°.
  - Wynik: `adsb_matches.csv` i PDF `air_<ICAO>_t<id>.pdf` (rejestracja, typ, lot, wysokość, odległość). Samoloty nie trafiają do `iod.txt`.
- **Ruch toru** (`tracks` rev 2): przyspieszenie z paraboli, zmienność prędkości między odcinkami 0,5 s, całkowita zmiana kierunku i największe tempo skrętu. W `tracks_final` jako `accel_deg_s2`, `speed_cv`, `turn_deg`, `turn_rate_deg_s`; w PDF niezidentyfikowanych.
- **„ptak?”:** istotna modulacja jasności w paśmie 2–15 Hz przy 0,5–20 °/s.
  - W DSCF4641 moc modulacji ≈ 3 ma prawie każdy tor, także satelity, więc to szum.
  - Istotną modulację (moc 19, 5,25 Hz) ma tylko #69.
  - PDF pokazuje teraz „brak istotnej (moc …)”, zamiast częstotliwości szumu.

## Weryfikacja i zgłaszanie: oświetlenie, classfd, IOD — 2026-09-28

- **Oświetlenie przez Słońce** jest teraz liczone także dla satelitów synchronizacji (`time_sync.json`) i dla przewidzianych w kadrze (`fov_predicted.csv`, tabela w `summary.pdf`). Wyjaśnia to, dlaczego część przewidzianych nie została wykryta. `identify` rev 2.
- **Katalog `classfd`** (Mike McCants): satelity spoza publicznych katalogów.
  - Pobierany jak CelesTrak do `cache/gp/`, konwersja TLE → OMM.
  - Dla torów bez dopasowania dodatkowe wyszukiwanie z szerszym oknem czasu (`identify.classfd_dt_tol_s`); wynik to kandydat z pewnością `low`.
  - `tle` rev 2.
- **Eksport IOD** do `report/iod.txt` (sekcja `iod`) do zgłoszeń w sieci SeeSat-L; `report` rev 2.
- **Migawka:** `camera.shutter_s: null` oznacza 1/fps z pliku (nowe nagrania: 29,97 kl/s, 1/30 s).

## Miejsce obserwacji per nagranie, paralaksa satelitów — 2026-09-28

**Przyczyna 0 identyfikacji w DSCF4641 (359 torów):**
- Nagranie było robione ~2,2 km od współrzędnych w `config.yaml`.
- Synchronizacja czasu wyszła poprawnie: Δ = +26,8 s, 21 zgodnych torów, Space-Track.
- Wszystkie tory miały jednak boczne odchylenie od orbit ∝ 1/odległość: 0,28° dla Starlinków (~480 km) i 0,06° dla Globalstara (~1740 km). Próg identyfikacji to 0,1°.
- Przykład: tor #58 to STARLINK-2112 (NORAD 47391).
  - zgodne: czas w kadrze, ω 0,771 vs 0,7709 °/s, kierunek 0,07°, 5″ wzdłuż toru;
  - niezgodne: tylko odchylenie boczne 0,26°.

**Zmiany:**
- **Współrzędne dla każdego nagrania.** Nowa komórka „Miejsce obserwacji” w notebooku.
  - Zapisuje je do `MyDrive/skyhunt/sites.yaml`, poza publicznym repo.
  - `load_config` wczytuje ten plik z `$SKYHUNT_SITES` i nakłada jako `files.<plik>.site`, z walidacją.
- **`identify` szacuje przesunięcie obserwatora z paralaksy torów synchronizacji.**
  - Metoda najmniejszych kwadratów; wynik trafia do `time_sync.json` → `observer_offset`.
  - Przy przesunięciu > `identify.site_warn_km` (0,3 km) pojawia się ostrzeżenie „sprawdź współrzędne”.
- **Klasyfikacja:** jasne obiekty (mediana `peak_snr` ≥ `classify.bright_peak_snr`) nie dostają etykiety „bliski obiekt?” z powodu szerokości, bo tę daje prześwietlenie.
- **Raport:** uwaga „bez Space-Track” pokazuje się tylko wtedy, gdy katalog faktycznie jest bez niego.

## Poprawki po pierwszym przebiegu na prawdziwych danych — 2026-09-28

- **`detect`:** limit komponentów na klatkę liczony tylko dla komponentów z maksimum ≥ `snr_seed`.
  - Wcześniej każda klatka była odrzucana, bo przekraczała limit komponentów rozrostu.
  - DSCF4641: 4,04 mln detekcji (mediana 522/klatkę), 0 klatek odrzuconych, 49 kl/s (L4).
  - Wynik: 359 torów.
- **`astrometry`:** `solve-field` uruchamiane z `--no-remove-lines --uniformize 0`.
  - Pomocnicze skrypty Pythona z pakietu apt ładowały NumPy 2 z Colaba i padały na `np.string_`.
  - Indeksy i pliki robocze są teraz kopiowane na dysk lokalny.
- **`dark_frames.MOV`:** `detect.sigma_floor_dn: 1.5` (≈ σ nieba).
  - H.264 kwantuje czerń, więc przy podłodze 0,5 DN 79% klatek było odrzucanych.

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
