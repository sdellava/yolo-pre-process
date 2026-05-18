# YOLO Person Detector

Progetto locale in Python che usa YOLO per identificare persone in immagini e salvare copie annotate con i bounding box.

## Requisiti

- Python 3.11+
- Connessione internet al primo avvio per scaricare il modello `yolov8n.pt` se non e' gia' presente in cache

## Installazione

Dal root del repository:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Uso

Metti una o piu' immagini dentro `input`, poi esegui:

```powershell
python detect_people.py --input input --output output
```

Per elaborare una singola immagine:

```powershell
python detect_people.py --input C:\percorso\immagine.jpg --output output
```

Per usare un modello `.pt` locale diverso:

```powershell
python detect_people.py --input input --output output --model C:\percorso\custom_model.pt
```

## Opzioni utili

```powershell
python detect_people.py --input input --output output --confidence 0.35 --model yolov8s.pt
```

- `--confidence`: soglia minima di confidenza
- `--model`: modello YOLO da usare (`yolov8n.pt`, `yolov8s.pt` o un file `.pt` locale)
- `--line-width`: spessore dei bounding box
- `--tile-imgsz`: dimensione di inferenza YOLO usata sui crop/ROI. Default: `960`
- `--min-zoom-width` e `--min-zoom-height`: se un crop e' piu' piccolo, viene ingrandito prima di passarlo a YOLO. Default: `640x480`

## Output

Per ogni immagine di input, lo script salva quattro file in `output`:

- `nomefile_A.jpg`: YOLO sull'immagine originale con bounding box
- `nomefile_B.jpg`: overlay della mappa di attenzione con i tile base e le ROI scelte dalla pre-analisi leggera
- `nomefile_C.jpg`: rilevazioni YOLO provenienti solo dai tile, riportate sull'immagine originale
- `nomefile_D.jpg`: merge finale dei box full-frame + tile riportato sull'immagine originale

Durante l'esecuzione, per ogni immagine vengono stampati anche i tempi in millisecondi:

- `orig_yolo`: latenza del solo YOLO sull'immagine originale
- `attention`: costruzione mappa di attenzione e scelta ROI
- `tile_yolo`: inferenza YOLO sui crop/ROI
- `merge`: fusione dei bounding box e disegno degli output
- `save`: scrittura dei quattro file di output
- `latency`: incremento assoluto e percentuale rispetto al solo `orig_yolo`

## Note

- Lo script disegna bounding box solo per la classe `person`.
- Se nell'immagine non ci sono persone, il file di output viene comunque salvato senza box.
- La prima esecuzione crea una cartella locale `.ultralytics` nel progetto per evitare problemi di permessi su Windows.
- Per migliorare il recall delle persone piccole, lo script esegue YOLO sia sul frame completo sia su tile sovrapposti ad alta risoluzione e poi fonde i box simili.
- Dopo YOLO full-frame, una pre-analisi leggera costruisce una mappa di attenzione basata su gradienti, bordi e micro-texture; le persone gia' trovate vengono attenuate nella mappa, cosi' i crop extra puntano alle zone ancora irrisolte.
- I crop piu' piccoli di `640x480` vengono zoomati prima dell'inferenza e analizzati con una `imgsz` piu' alta, in modo che persone lontane occupino piu' celle utili nelle feature map di YOLO.
- Le detection troppo vicine al bordo interno di un tile vengono scartate per ridurre box tagliati e duplicati.
