# HANDOFF → Claude Code: `skyhunt` — detekcja słabych i nietypowych obiektów ruchomych na wideo nieba (Colab Pro, GPU)

Język projektu: komentarze, README i raporty po polsku. Nazwy w kodzie po angielsku.
Styl: konkretnie, bez lania wody. Każda heurystyka musi mieć parametr w configu i test.

---

## 1. Cel

Zbudować pipeline, który z wielu nagrań nieba (pliki ~1 GB, 4K) wyciąga **wszystkie** obiekty ruchome, także te na granicy szumu. Każdy obiekt pipeline ma:

1. zmierzyć (astrometria, kinematyka, fotometria, kształt PSF),
2. **wyjaśnić**, jeśli się da, czyli dopasować do satelity z TLE, meteoru, samolotu z ADS-B albo bliskiego obiektu (ptak, nietoperz, owad),
3. oznaczyć jako **anomalię** tylko wtedy, gdy po odrzuceniu znanych klas coś zostaje, i to według kryteriów ustalonych *przed* analizą (sekcja 7).

Czułość bez kontroli fałszywych alarmów jest bezwartościowa. Pipeline musi raportować **limit detekcji** (testy injection–recovery) i **oczekiwaną liczbę fałszywych detekcji** na godzinę nagrania.

---

## 2. Dane i sprzęt

- Aparat: Fujifilm X-E3, obiektyw 50 mm f/1.0, statyw, ostrość na nieskończoność.
- Wideo: 3840×2160, 23.976 fps (2997/125), H.264 (MP4, ~100 Mb/s), pliki ~1 GB (~80 s), audio AAC (ignorować).
- Miejsce: okolice Łodzi. **Dokładne współrzędne i wysokość n.p.m. podaje użytkownik w `config.yaml`.**
- Czas: tag `creation_time` z MP4 (np. `2026-09-27T19:30:34Z`). **Uwaga:** Fuji zapisuje czas lokalny aparatu oznaczony jako UTC. Czasu nie wolno ufać. Trzeba go wyznaczyć przez dopasowanie przelotów satelitów (sekcja 5.6).
- Pole widzenia: nominalnie ~26.4°×14.9° (APS-C 23.5 mm, 16:9). **Nie wiadomo, czy X-E3 przycina kadr w 4K.** Rozstrzyga plate solving; nie wolno zakładać z góry.
- Pliki leżą na Google Drive użytkownika (`/content/drive/MyDrive/skyhunt/raw/`).

## 3. Co już wiadomo (analiza pilotażowa, plik `DSCF4641 - Trim2.mp4`, 7.5 s)

Baseline CPU jest w `skytracks.py` (w załączniku). Traktuj go jako referencję wyników, nie architektury.

- 5 torów ~0.88–0.91°/s: proste (odchyłka od prostej < 0.4 px przy 1920 px), podobny kierunek. To satelity LEO, prawdopodobnie jedna powłoka orbitalna.
- 1 tor 4.44°/s: zakrzywiony (13 px odchyłki), szerokość poprzeczna PSF ~2× gwiazdy, czyli **nieostry, a więc bliski**. Jasność pulsuje z pikiem 5.2 Hz i harmoniczną 10.5 Hz; możliwy aliasing, prawdziwa częstotliwość to 5.2 albo 18.8 Hz. Z rozmycia odległość wychodzi ~50–100 m, prędkość ~4–8 m/s. **Wniosek: nietoperz lub mały ptak.** To przypadek testowy klasy „bliski obiekt”.
- Na stacku max−tło widać też bardzo słabe liniowe ślady, których baseline nie wykrył. To główny cel nowego pipeline'u.

---

## 4. Środowisko: Colab Pro

- GPU: A100 lub L4 (sprawdź `nvidia-smi`). Wszystko ciężkie w PyTorch/CuPy na GPU.
- Struktura: repo Pythonowe (pakiet `skyhunt/`) + notebook `colab/run_skyhunt.ipynb`, który montuje Drive, instaluje zależności, robi `git clone`/`pip install -e .` i uruchamia CLI.
- **Odporność na rozłączenia Colaba:** przetwarzanie per plik, wyniki pośrednie na Drive, wznawianie od ostatniego gotowego etapu (`manifest.json` per plik: etap, hash configu, wersja kodu).
- Dekodowanie wideo na GPU w tej kolejności prób:
  1. `PyNVVideoCodec` / `torchcodec` (NVDEC), klatki od razu jako tensory CUDA,
  2. `torchaudio.io.StreamReader` z `h264_cuvid`,
  3. fallback: PyAV/ffmpeg na CPU w wątku producenta + pinned memory.

  Zmierz przepustowość każdej opcji i zapisz w README.
- Budżet pamięci: 4K gray float16 to ~16 MB na klatkę. Cały plik (~1900 klatek) nie mieści się komfortowo, więc przetwarzaj w **oknach czasowych z zakładką** (np. 256 klatek, zakładka 64).
- Zależności (propozycja, zweryfikuj dostępność): `torch`, `cupy-cuda12x`, `numpy`, `scipy`, `astropy`, `photutils`, `skyfield`, `sgp4`, `tetra3` lub `cedar-solve` (szybki plate solve), opcjonalnie lokalny `astrometry.net` (`solve-field` z indeksami 4100 dla pól ~15–30°), `pandas`/`pyarrow`, `plotly`, `imageio[ffmpeg]`, `pyyaml`, `rich`.

---

## 5. Architektura pipeline'u

```
raw.mp4
 └─ 5.1 dekodowanie GPU (Y z YUV, pełna rozdzielczość) + metadane
 └─ 5.2 model tła i szumu per piksel (okno kroczące)
 └─ 5.3 maska gwiazd / hot pikseli / artefaktów H.264
 └─ 5.4 detekcja: (a) klasyczna per klatka, (b) synthetic tracking / shift-and-stack, (c) 3D (x,y,t)
 └─ 5.5 łączenie w tory + kinematyka + fotometria + PSF
 └─ 5.6 astrometria (plate solve → WCS, dystorsja) i synchronizacja czasu po satelitach
 └─ 5.7 identyfikacja: TLE / meteory / ADS-B / bliskie obiekty
 └─ 5.8 scoring anomalii
 └─ 5.9 raport: katalog Parquet/CSV, wycinki GIF/MP4, dashboard HTML
```

### 5.1 Dekodowanie
- Używaj kanału Y (luminancja) w pełnej rozdzielczości 4K. Chroma opcjonalnie, do koloru obiektów w 5.5.
- Zapisz typy klatek (I/P/B) i pozycje GOP. Są potrzebne do maskowania artefaktów kompresji: pulsowanie jasności co GOP i bloki 16×16.

### 5.2 Tło i szum
- Tło: mediana (lub średnia obcięta sigma-clippingiem) per piksel w oknie kroczącym ±N klatek (np. N=48), liczona na GPU (`torch.median` na osi czasu w kaflach).
- Szum: odporna sigma per piksel (MAD) w tym samym oknie. Detekcja działa na mapie **SNR = (klatka − tło) / σ**, nie na surowych DN.
- Uwzględnij powolny ruch gwiazd (rotacja nieba ~15″/s, przy 50 mm to ~1 px 4K na ~4 s). Okno tła musi być na tyle krótkie, żeby gwiazdy nie zostawiały „duchów” w różnicy, albo trzeba robić rejestrację klatek do WCS przed różnicowaniem. **Preferowane: odejmowanie po rejestracji na gwiazdy** (shift per klatka z plate solve lub z korelacji fazowej).

### 5.3 Maski i artefakty (obowiązkowe, każdy z testem)
- Hot/warm pixele: stałe w czasie, wysoki SNR, 1 px.
- Scyntylacja gwiazd: fluktuacje w miejscu gwiazdy, maska z katalogu gwiazd po plate solve (promień ∝ jasność).
- H.264: krawędzie makrobloków, pulsowanie przy I-klatkach, „pływanie” szumu w blokach P/B. Detekcja, która pojawia się tylko na I-klatkach albo jest wyrównana do siatki 16 px, ma być odrzucona lub oflagowana.
- **Duchy optyczne (ghosty):** jasny obiekt ruchomy może mieć odbicie symetryczne względem środka optycznego, poruszające się „dziwnie”. Sprawdzaj pary torów symetrycznych punktowo.
- Chmury i zmiany tła nieba: flaga globalna na okno.
- Owady bardzo blisko obiektywu: ogromne rozmycie (bokeh-dyski), często prześwietlone. Klasa „bliski obiekt”.

### 5.4 Detekcja (serce projektu)
Trzy równoległe detektory; wyniki łączone i deduplikowane.

**(a) Klasyczna per klatka.** Matched filter PSF na mapie SNR, próg ~5σ, etykietowanie spójnych obszarów. Łapie jasne obiekty i meteory.

**(b) Synthetic tracking / shift-and-stack (track-before-detect).** Dla siatki hipotez prędkości (vx, vy) w px/klatkę sumuj mapy SNR przesunięte o (vx·t, vy·t) w krótkich podoknach (K = 8–48 klatek). Obiekt poruszający się liniowo zyskuje SNR ∝ √K: przy K=24 daje to ~5× i ok. 1.7 mag głębiej.
- Siatka prędkości od ~0.05 do ~30 px/klatkę (4K), gęstość dobrana tak, żeby rozmycie z niedopasowania < 1 FWHM przez K klatek.
- Implementacja na GPU: batched `torch.roll`/`grid_sample` albo FFT (przesunięcie = mnożenie fazą). Kafelkowanie po obrazie.
- Obiekty szybkie (> kilka px/klatkę) są w klatce rozmazane w kreskę. Filtr musi być kreską dopasowaną do (vx, vy) i czasu ekspozycji (przyjmij migawkę ~1/24–1/50 s; wyznacz z długości smug satelitów).
- Kontrola trials factor: liczba hipotez jest ogromna, więc próg dobierz z rozkładu wyników na **danych z przetasowaną kolejnością klatek** (tło bez spójnego ruchu). To daje empiryczny FAR.

**(c) Detekcja 3D (x, y, t) dla ruchu nieliniowego.** Tor zakrzywiony, przyspieszający lub zygzakowaty nie zostanie w pełni zsumowany przez (b). Opcje do porównania:
- krótkie segmenty z (b) (K małe) łączone potem w tory nieliniowe w 5.5,
- 3D Hough/Radon lokalnie,
- opcjonalnie lekki model uczony (3D U-Net na kostkach 64×64×32) trenowany **wyłącznie na syntetycznych wstrzyknięciach** z realnym szumem z tych nagrań. Nie jest wymagany w MVP.

### 5.5 Tory i pomiary
- Łączenie: Kalman (stała prędkość) + dopuszczenie modelu stałego przyspieszenia; gating na podstawie niepewności; łączenie przerw do kilku klatek.
- Per tor zapisz:
  - pozycje (px i RA/Dec lub Az/Alt po 5.6), prędkość kątowa [°/s], przyspieszenie, krzywizna (residuum od prostej i od koła wielkiego na sferze),
  - czas trwania; czy tor zaczyna i kończy się **wewnątrz** kadru (pojawienie/zniknięcie) czy na krawędzi,
  - fotometria (strumień w aperturze dopasowanej do smugi, kalibracja zero-pointu z gwiazd, czyli magnitudo), krzywa jasności, **widmo jasności** (FFT/Lomb-Scargle) z listą pików i ich aliasów względem 23.976 fps,
  - PSF: szerokość poprzeczna do ruchu ÷ szerokość gwiazd w tym samym miejscu kadru (dystorsja!); kształt (dysk bokeh vs gauss),
  - kolor (Cb/Cr) jako słaba cecha.
- **Odległość z rozmycia:** przy ostrości na ∞ kątowa średnica krążka rozmycia θ ≈ D/d, gdzie D = f/N = 50 mm. Zaimplementuj estymator z dekonwolucją PSF gwiazdy i niepewnością; wynik: d i prędkość liniowa v = ω·d. Tylko dla obiektów wyraźnie nieostrych; dla ostrych podawaj dolną granicę d.

### 5.6 Astrometria i czas
- Plate solve na klatce uśrednionej (tetra3/cedar szybko, astrometry.net dokładnie). Wynik: WCS z SIP (dystorsja obiektywu f/1.0 będzie znacząca na brzegach), **rzeczywiste pole widzenia** (rozstrzyga crop 4K).
- Mając WCS, pozycję obserwatora i przybliżony czas: przelicz Az/Alt.
- **Synchronizacja czasu:** tory proste o prędkościach typowych dla LEO dopasuj do przewidywań Skyfield z TLE (CelesTrak, archiwalne TLE z dnia nagrania). Minimalizuj rezydua pozycji po przesunięciu czasu Δt (szukaj w ±2 h, potem dokładnie). Wynik: poprawka zegara aparatu (w tym ewentualny błąd strefy czasowej) z niepewnością. Zapisz ją per plik i per sesję.

### 5.7 Identyfikacja znanych klas
- **Satelity:** po synchronizacji czasu każdy tor dopasuj do katalogu (prędkość, kierunek, pozycja, czas; tolerancje w configu). Oznacz NORAD ID, nazwę, wysokość, czy jest oświetlony przez Słońce (Skyfield: cień Ziemi). **Obiekt poruszający się jak satelita, ale w cieniu Ziemi, to istotna flaga.**
- **Meteory:** krótkie (< ~2 s), proste, szybkie, często z rozbłyskiem. Sprawdź zgodność z radiantami aktywnych rojów (lista IMO na datę) przez przedłużenie toru na sferze.
- **Samoloty:** błyski stroboskopowe ~0.5–1.5 Hz, światła pozycyjne. Opcjonalnie dopasowanie do historycznego ADS-B (np. adsb.lol / OpenSky, jeśli dostęp) po czasie i azymucie.
- **Bliskie obiekty biologiczne:** nieostrość > próg, modulacja jasności 3–25 Hz (z uwzględnieniem aliasingu), tory nieliniowe, prędkości liniowe po estymacji odległości 2–20 m/s.
- Każda identyfikacja ma pole `confidence` i `reason`.

### 5.8 Scoring anomalii
- Anomalia to tor **niewyjaśniony** przez 5.7, który przeszedł wszystkie maski z 5.3 i jest wykryty w ≥ 2 niezależnych detektorach lub z SNR powyżej progu FAR.
- Wynik liczbowy: np. odległość Mahalanobisa w przestrzeni cech od chmury torów wyjaśnionych + reguły twarde, np. nagła zmiana kierunku przy ostrym PSF, prędkość kątowa > 2°/s przy ostrym PSF i czasie > 2 s, pojawienie/zniknięcie wewnątrz kadru bez przejścia w cień Ziemi.
- Dla każdej anomalii automatycznie wygeneruj **listę zwyczajnych hipotez, których nie dało się sprawdzić**, np. brak ADS-B albo brak drugiej stacji.

### 5.9 Wyjścia
Per plik:
- `tracks.parquet` + `tracks.csv` (wszystkie cechy, klasa, confidence, reason, anomaly_score),
- `stack_annotated.png` (4K),
- `cutouts/<track_id>.mp4|gif`: wycinek wideo wokół toru, surowy i SNR, plus wersja shift-and-stack w układzie obiektu,
- `lightcurves/<track_id>.png` z widmem jasności.

Per sesja:
- `report.html`: tabela torów z filtrami, statystyki klas, limit detekcji (mag vs prędkość), FAR, poprawka czasu, lista anomalii z wycinkami.

---

## 6. Walidacja (obowiązkowa przed interpretacją wyników)

1. **Injection–recovery:** wstrzykuj syntetyczne obiekty z PSF zmierzonym z gwiazd (liniowe, zakrzywione, pulsujące, rozmyte, o różnych mag i prędkościach) w realne nagrania. Mapa kompletności: magnitudo × prędkość kątowa × krzywizna.
2. **FAR:** uruchom pełny pipeline na nagraniach z przetasowaną kolejnością klatek i na nagraniach z zakrytym obiektywem (poproś użytkownika o 1–2 takie pliki: dark frames wideo).
3. **Regresja na pliku pilotażowym:** 5 satelitów ~0.9°/s i 1 bliski obiekt 4.4°/s z pikiem 5.2 Hz mają wyjść poprawnie (test w `tests/`).
4. Sprawdź, czy liczba wykrytych satelitów jest zgodna z przewidywaniem Skyfield dla pola widzenia i czasu (po korekcie jasności i oświetlenia).

## 7. Zasady interpretacji (wpisać do README i raportu)

- Kryteria „anomalii” są zamrożone w `config.yaml` przed przetworzeniem pełnego zbioru. Zmiana kryteriów wymaga nowej wersji i ponownego przeliczenia całości.
- Jedna kamera nie daje odległości ani prędkości liniowej, z wyjątkiem estymacji z rozmycia dla obiektów bliskich. Raport nigdy nie podaje km/s dla ostrych obiektów bez triangulacji.
- Rekomendacja sprzętowa do raportu: druga kamera w odległości 5–20 m (paralaksa rozstrzyga bliski/daleki natychmiast); nagrania 1080p/60 fps obok 4K/24 fps (rozstrzyga aliasing trzepotu); okresowo 1–2 min z zakrytym obiektywem.

## 8. Plan realizacji (kamienie milowe)

1. **M0:** repo, config, notebook Colab, dekodowanie GPU z pomiarem przepustowości, manifest/wznawianie.
2. **M1:** tło+szum+maski, detektor (a), tory, regresja na pliku pilotażowym (parytet z `skytracks.py`).
3. **M2:** plate solve + WCS + synchronizacja czasu po satelitach + identyfikacja TLE.
4. **M3:** detektor (b) shift-and-stack na GPU + FAR z tasowania + injection–recovery.
5. **M4:** tory nieliniowe (c), estymacja odległości z rozmycia, klasyfikacja biologiczna, meteory, samoloty.
6. **M5:** scoring anomalii, raport HTML, wycinki.

Po każdym kamieniu: krótkie podsumowanie w `CHANGELOG.md` (co działa, liczby: przepustowość, limit mag, FAR) i pytanie do użytkownika, zanim ruszysz dalej, jeśli wynik zmienia założenia.

## 9. Otwarte pytania do użytkownika (zadać na starcie)
1. Dokładne współrzędne i wysokość miejsca obserwacji; czy wszystkie nagrania z jednego miejsca?
2. Model obiektywu 50 mm f/1.0; czy był ustawiony na ∞ i czy ostrość była sprawdzana na gwiazdach?
3. Ustawienia migawki i ISO w trybie wideo (jeśli znane).
4. Strefa czasowa ustawiona w aparacie.
5. Ile plików / godzin nagrań i gdzie dokładnie leżą na Drive.
