# Sound Wave 🎵

Un equalizzatore grafico audio in tempo reale per desktop, con un pannello di controllo moderno e una visualizzazione OpenGL a finestra senza bordi ad alte prestazioni.

![Python](https://img.shields.io/badge/python-v3.12+-blue.svg)
![Platform](https://img.shields.io/badge/platform-windows%20%7C%20linux%20%7C%20macos-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

> 🇬🇧 **English**: [Read this README in English](README.md)

## 📋 Panoramica

Sound Wave cattura l'audio di sistema in tempo reale, ne esegue l'analisi FFT e disegna visualizzazioni animate delle frequenze. È composto da due parti che cooperano:

- Un **pannello di controllo PySide6** dove scegli il dispositivo audio e configuri ogni aspetto dell'aspetto grafico.
- Un unico **renderer GLFW/OpenGL a finestra senza bordi** che copre il monitor scelto. *Non* è fullscreen esclusivo — l'alt-tab è istantaneo e non c'è cambio di modalità video. Il renderer può disegnare un'immagine o un video in loop (muto) dietro le barre.

Le configurazioni si possono salvare come **profili con nome** e condividere come file, così non devi più impostare tutto da zero ogni volta.

### ✨ Caratteristiche principali

- **🎤 Cattura audio multi-dispositivo** — cattura l'audio di sistema (44.1 kHz, stereo) da qualsiasi dispositivo disponibile.
- **📊 Quattro modalità di visualizzazione** — **Barre**, **Radiale**, **Oscilloscopio**, **Linea/Area**.
- **🎨 Quattro modalità colore** — **Classico** (verde/giallo/rosso), **Tinta unica**, **Gradiente**, **Spettro** (HSV lungo l'asse delle frequenze).
- **🎛️ Controlli di forma** — larghezza barre, cime arrotondate, ancoraggio al centro, e ordine bande simmetrico "centrato sui bassi".
- **✨ Effetti reattivi** — **peak-cap** per-barra, **glow (bloom)**, e un **beat pulse** che zooma lo sfondo sui beat rilevati.
- **🖼️ Sfondi** — **immagine** personalizzata *oppure* **video** in loop (puramente decorativo; dal video non viene decodificato alcun audio).
- **💾 Gestore profili** — salva/carica profili con nome, **importa/esporta** come `.json`, ricarica automaticamente l'ultimo profilo all'avvio, con un **safe fallback** quando un file di sfondo referenziato non si trova.
- **🔬 DSP avanzato** — scala dB assoluta con pavimento/tetto regolabili, ballistica attacco/rilascio, e un controllo di tilt spettrale.
- **🌗 Temi** — chiaro / scuro / sistema, commutabili a caldo.
- **🖥️ Multi-monitor** — scegli quale display copre il renderer.

## 🖼️ Screenshot

### Pannello di controllo
Il pannello PySide6: in alto il selettore dispositivo e la barra profili, una sidebar di navigazione (Visualizzazione / Forma / Colore / Audio / Effetti) che pilota le pagine di impostazioni.

### Renderer fullscreen
Visualizzazione OpenGL immersiva che copre il monitor, con VSync, antialiasing MSAA 4× e rendering instanced.

## 🛠️ Installazione

### Requisiti di sistema

- **Python**: 3.12 o superiore
- **Sistema operativo**: Windows, macOS, Linux
- **Hardware**: un dispositivo audio e una GPU/driver con supporto OpenGL 3.3 core
- **Extra runtime Linux**: PulseAudio/PipeWire (per `soundcard`) e un display con GL; `libxcb-cursor0` per il plugin di piattaforma xcb di Qt

### Dipendenze

Il progetto è gestito con [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + il `uv.lock` bloccato, `requires-python = ">=3.12"`). **Non** esiste un `requirements.txt`.

```bash
uv sync          # crea/aggiorna l'ambiente virtuale dal lockfile
```

Dipendenze principali:
- `PySide6` — la GUI del pannello di controllo (Qt for Python, LGPL)
- `pyqtdarktheme-fork` (importato come `qdarktheme`) — temi chiaro/scuro/sistema
- `soundcard` — cattura dell'audio di sistema cross-platform
- `numpy` — FFT e DSP vettorizzato
- `PyOpenGL` + `PyOpenGL-accelerate` — rendering OpenGL (devono risolvere alla **stessa** versione)
- `glfw` — gestione finestra/contesto OpenGL
- `Pillow` — gestione immagini
- `imageio` + `imageio-ffmpeg` — decodifica dei video di sfondo (include un `ffmpeg` statico)

### Avvio rapido

1. **Clona il repository**:
   ```bash
   git clone https://github.com/Cioscos/easy-graphic-equalizer.git
   cd easy-graphic-equalizer
   ```

2. **Installa le dipendenze**:
   ```bash
   uv sync
   ```

3. **Avvia l'applicazione**:
   ```bash
   uv run python main.py
   ```

## 🚀 Utilizzo

### Per iniziare

1. **Seleziona un dispositivo audio** dalla lista in alto. Il pulsante d'azione in basso diventa verde ("Avvia a schermo intero") una volta selezionato un dispositivo.
2. **Configura l'aspetto** usando le pagine di impostazioni (vedi sotto) — le modifiche si applicano a caldo mentre il renderer è in esecuzione.
3. **Avvia il renderer** con il pulsante in basso. Si apre una finestra senza bordi che copre il monitor selezionato e il pulsante diventa rosso ("Ferma").
4. **Esci dal renderer** premendo **ESC** (o fermandolo dal pannello di controllo).

### Pagine di impostazioni

La sidebar di navigazione commuta tra cinque pagine:

- **🖥️ Visualizzazione** — modalità di visualizzazione, sfondo (immagine/video) + trasparenza, opacità barre, monitor di destinazione.
- **🎛️ Forma** — larghezza barre, cime arrotondate, ancoraggio, ordine bande; più i controlli di forma specifici della modalità (raggio del foro radiale, spessore linea, riempimento, specchiatura/fluidità oscilloscopio).
- **🎨 Colore** — modalità colore e i relativi colori/soglie (mostrati condizionalmente per modalità).
- **🎵 Audio** — soglia rumore, numero di bande di frequenza, e un gruppo **Avanzate (DSP)** (pavimento/tetto dB, attacco, rilascio, tilt spettrale).
- **✨ Effetti** — peak-cap, glow (bloom) e beat-pulse (tutti **off** di default).

### ⌨️ Scorciatoie da tastiera

- **ESC** — esci dal renderer fullscreen.

## 💾 Profili (import / export)

Un **profilo** cattura l'intera configurazione visiva/DSP/audio/sfondo (esclude di proposito gli elementi specifici della macchina come dispositivo audio, monitor e tema dell'app).

La **barra profili** in cima al pannello di controllo permette di:

- **Selezionare** un profilo salvato dal menu a tendina per applicarlo subito (a caldo se il renderer è in esecuzione).
- **Salva** — sovrascrive il profilo selezionato con le impostazioni correnti.
- **Salva come…** — memorizza le impostazioni correnti con un nuovo nome.
- **Elimina** — rimuove il profilo selezionato.
- **Importa…** — carica un profilo `.json` da qualsiasi posizione (i nomi vengono deduplicati automaticamente).
- **Esporta…** — scrive un profilo su un file `.json` per fare backup o condividerlo.

L'ultimo profilo selezionato viene **ricaricato automaticamente** al lancio successivo.

### Archiviazione e safe fallback

I profili sono salvati come un file JSON ciascuno in una cartella di config per-utente:

| OS | Posizione |
|----|-----------|
| Windows | `%APPDATA%\sound-wave\profiles\` |
| macOS | `~/Library/Application Support/sound-wave/profiles/` |
| Linux | `$XDG_CONFIG_HOME/sound-wave/profiles/` (oppure `~/.config/sound-wave/profiles/`) |

I riferimenti agli sfondi sono **portabili**: uno sfondo incluso nell'app viene salvato come riferimento-risorsa (così si risolve su qualsiasi macchina e in una build pacchettizzata), mentre un file esterno viene salvato come path assoluto. Se uno sfondo referenziato **non si trova** all'applicazione di un profilo, Sound Wave mostra un avviso non bloccante e ricade sull'immagine predefinita inclusa — non va mai in crash per un file mancante.

## 🎛️ Modalità di visualizzazione

| Modalità | Descrizione |
|----------|-------------|
| **Barre** | Barre di frequenza instanced classiche (supportano peak-cap, cime arrotondate, ancoraggio al centro). |
| **Radiale** | Le barre avvolte in un anello polare con raggio interno regolabile. |
| **Oscilloscopio** | La forma d'onda (mediata sui canali), allineata su zero-crossing, con specchiatura opzionale e smoothing temporale. |
| **Linea/Area** | Uno spettro a linea continua, opzionalmente riempito ad area con una sfumatura alpha verticale. |

L'ordine bande simmetrico "centrato sui bassi" si applica a Barre / Radiale / Linea. Il peak-cap è solo per le Barre.

## ✨ Effetti reattivi (off di default)

- **Peak-cap** — un cappuccio che scende e marca il picco recente di ogni barra (colore, hold e velocità di caduta configurabili).
- **Glow (bloom)** — un bloom gaussiano a metà risoluzione composto in modo additivo sopra la visualizzazione (funziona in ogni modalità).
- **Beat pulse** — rileva i beat dall'energia FFT delle basse frequenze e pilota un inviluppo di zoom dello sfondo.

## 🔬 Dettagli tecnici

### Elaborazione audio

- **Frequenza di campionamento**: 44.1 kHz, **stereo** (2 canali), catturata in chunk da 1024 frame.
- **Dimensione FFT**: 4096 campioni, ricalcolata **a ogni frame** sulla finestra più recente da un ring buffer.
- **Funzione finestra**: Blackman (riduce lo spectral leakage).
- **FFT vettorizzata a passata singola**: una `numpy.rfft` su entrambi i canali in una volta; i bin sono aggregati in bande tramite una mappa `bin→banda` precalcolata. **Niente** multiprocessing — una rFFT a 4096 punti è ~0.1 ms, quindi una FFT seriale per frame è più economica dell'IPC.
- **Scala dB assoluta**: potenza per banda → dB → mappata da un intervallo fisso `[DB_FLOOR, DB_CEILING]` a `[0, 1]` (l'altezza della barra riflette il volume assoluto, non un massimo per frame).
- **Ballistica**: smoothing **attacco/rilascio** indipendente dal frame rate (salita rapida, discesa morbida) verso i target FFT.

### Rendering

- **OpenGL 3.3 core profile** con shader, antialiasing **MSAA 4×**.
- Output **a finestra senza bordi** dimensionato sul monitor selezionato (`GLFW_DECORATED=FALSE` + posizionamento finestra) — alt-tab istantaneo, nessun cambio di modalità.
- **Bloccato in VSync** sulla frequenza di refresh del monitor.
- Le barre usano **rendering instanced** (un VBO statico unit-quad + un VBO di altezze per-istanza); cambiare il numero di bande ridimensiona solo il buffer delle altezze.
- Lo sfondo è un quad fullscreen texturizzato la cui alpha è una uniform dello shader; i frame **video** sono trasmessi via 2 **PBO** ping-pong (upload DMA asincrono) e decodificati su un thread dedicato, ritmati alla frequenza reale del video.

## 📁 Struttura del progetto

```
easy-graphic-equalizer/
├── main.py                          # Entry point (costruisce la QApplication)
├── pyproject.toml                   # Metadati progetto e dipendenze (uv)
├── uv.lock                          # Lockfile delle dipendenze (uv)
├── sound-wave.spec                  # Spec PyInstaller cross-platform
├── profiles.py                      # Modello profilo + ProfileStore (persistenza, import/export)
├── resource_manager.py              # Risoluzione path risorse (frozen-aware)
│
├── gui/                             # Pannello di controllo PySide6
│   ├── main_window_gui.py           # Finestra principale, thread/code, collect/apply profilo
│   ├── profile_bar_frame.py         # Barra selettore profili + azioni
│   ├── animated_button.py           # Pulsante d'azione animato
│   ├── color_picker_frame.py        # Color picker + preset
│   ├── background_filepicker_frame.py # Selettore file immagine/video
│   ├── slider_frame.py              # Componente slider custom
│   ├── optionmenu_frame.py          # Componente menu a tendina
│   ├── help_window.py               # Finestra di aiuto
│   └── theme.py                     # Token di stile, font, strato qdarktheme
│
├── thread/                          # Pipeline produttore/consumatore
│   ├── audioCaptureThread.py        # Produttore cattura audio
│   ├── opengl_thread.py             # Renderer OpenGL (unico consumatore)
│   ├── video_decode_thread.py       # Produttore opzionale decodifica video
│   ├── AudioBufferAccumulator.py    # Ring buffer per le finestre FFT
│   └── frame_time_stats.py          # Statistiche frame-time (debug)
│
├── tests/                           # Verifiche Python pure (niente pytest)
│   ├── test_profiles.py             # Test serializzazione/store profili
│   └── test_frame_time_stats.py     # Test statistiche frame-time
│
└── resources/                       # Asset dell'applicazione
    ├── bg/                          # Immagini di sfondo
    └── icons/                       # Icone dell'interfaccia
```

## 📦 Creazione di una release

L'app è pacchettizzata con **PyInstaller** usando l'unico spec cross-platform:

```bash
uv run --with pyinstaller pyinstaller --noconfirm sound-wave.spec
# output: dist/sound wave/   (bundle one-dir)
```

Il workflow GitHub Actions `.github/workflows/release-build.yml` costruisce il bundle su **Windows** e **Linux** e allega `sound-wave-windows.zip` / `sound-wave-linux.zip` a una release pubblicata. PyInstaller non cross-compila, da cui un runner per OS. La cartella `resources/` viene inclusa nella build, e `ResourceManager` risolve i path degli asset da `sys._MEIPASS` quando è frozen.

## 🐛 Risoluzione dei problemi

#### L'app non rileva i dispositivi audio
- Verifica che i driver audio siano aggiornati.
- Controlla i permessi di microfono/registrazione (macOS/Linux).
- Su Linux, assicurati che PulseAudio/PipeWire sia in esecuzione.

#### Il renderer non parte / non mostra nulla
- Verifica che i driver della GPU siano aggiornati e supportino OpenGL 3.3 core.
- Su Linux, assicurati che le librerie GL e un display con GL siano disponibili.

#### Problemi di prestazioni con molte bande
- Riduci il numero di bande di frequenza.
- Disabilita gli effetti pesanti (bloom) se la GPU è modesta.

#### Un'immagine/video di sfondo non si carica
- Verifica il formato (immagini: PNG/JPG/JPEG/GIF; video: MP4/AVI/MOV/MKV).
- Controlla i permessi di lettura e che il path esista ancora.
- Per i profili, uno sfondo mancante ricade sul predefinito con un avviso.

### Logging e debug

Abilita il logging dettagliato tramite i flag a livello di modulo in cima a `thread/opengl_thread.py`:

```python
GENERAL_LOG = True   # log generali
FFT_LOGS = True      # log elaborazione FFT
PERF_LOGS = True     # riepiloghi aggregati dei frame-time
```

## 🤝 Contribuire

I contributi sono benvenuti!

1. Fai un **fork** del repository
2. Crea un **branch** per la tua feature (`git checkout -b feature/nuova-feature`)
3. **Committa** le tue modifiche
4. **Pusha** sul branch
5. Apri una **Pull Request**

### Linee guida

- Segui lo stile e i pattern del codice esistente.
- Aggiungi docstring per nuove funzioni/classi.
- Mantieni allineato il DSP se tocchi la logica FFT/bande di frequenza.
- Aggiorna la documentazione quando il comportamento cambia.

## 📄 Licenza

Rilasciato sotto licenza MIT. Vedi il file `LICENSE` per i dettagli.

## 🙏 Ringraziamenti

- **PySide6 / Qt** — GUI del pannello di controllo
- **pyqtdarktheme-fork** — temi
- **SoundCard** — cattura audio cross-platform
- **NumPy** — FFT e DSP
- **PyOpenGL / GLFW** — rendering ad alte prestazioni
- **imageio / FFmpeg** — decodifica dei video di sfondo

---

**Sviluppato con ❤️**
