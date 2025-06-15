# Sound Wave 🎵

Un equalizzatore grafico audio in tempo reale con interfaccia moderna e visualizzazione fullscreen ad alte prestazioni.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Platform](https://img.shields.io/badge/platform-windows%20%7C%20linux%20%7C%20macos-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

## 📋 Panoramica

Sound Wave è un'applicazione desktop che cattura l'audio in tempo reale dai dispositivi di input del sistema e lo visualizza come un equalizzatore grafico dinamico. L'applicazione offre sia una modalità di anteprima integrata che una modalità fullscreen ad alte prestazioni per monitor multipli.

### ✨ Caratteristiche Principali

- **🎤 Cattura Audio Multi-dispositivo**: Supporta tutti i dispositivi audio disponibili sul sistema
- **📊 Visualizzazione Tempo Reale**: Equalizzatore con barre colorate (verde, giallo, rosso)
- **🖥️ Modalità Fullscreen**: Visualizzazione ad alte prestazioni con supporto multi-monitor
- **⚙️ Configurazione Avanzata**: 
  - Soglia di rumore regolabile
  - Numero di bande di frequenza dinamico (1-100)
  - Trasparenza dello sfondo personalizzabile
  - Elaborazione parallela configurabile
- **🎨 Personalizzazione Visiva**:
  - Supporto immagini di sfondo personalizzate
  - Temi chiari/scuri/automatici
  - Controllo dell'alpha del background
- **⚡ Ottimizzazioni delle Prestazioni**:
  - Calcolo FFT parallelo multi-core
  - Rendering OpenGL per la modalità fullscreen
  - Smoothing adattivo basato sul refresh rate del monitor

## 🖼️ Screenshots

### Interfaccia Principale
L'interfaccia principale con controlli sulla sinistra e anteprima dell'equalizzatore sulla destra.

### Modalità Fullscreen
Visualizzazione immersiva a schermo intero con rendering OpenGL ad alte prestazioni.

## 🛠️ Installazione

### Requisiti di Sistema

- **Python**: 3.8 o superiore
- **Sistema Operativo**: Windows, macOS, Linux
- **Hardware**: Scheda audio con driver compatibili

### Dipendenze

```bash
pip install -r requirements.txt
```

Le dipendenze principali includono:
- `customtkinter` - Interfaccia utente moderna
- `soundcard` - Cattura audio cross-platform
- `numpy` & `scipy` - Elaborazione numerica e FFT
- `PyOpenGL` - Rendering grafico ad alte prestazioni
- `Pillow` - Gestione immagini
- `glfw` - Gestione finestre OpenGL

### Avvio Rapido

1. **Clona la repository**:
   ```bash
   git clone https://github.com/tuousername/sound-wave.git
   cd sound-wave
   ```

2. **Installa le dipendenze**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Avvia l'applicazione**:
   ```bash
   python main.py
   ```

## 🚀 Utilizzo

### Configurazione Iniziale

1. **Seleziona un Dispositivo Audio**: 
   - Fai doppio clic su un dispositivo dalla lista sulla sinistra
   - Il pulsante "Start/Pause" diventerà verde quando pronto

2. **Avvia la Visualizzazione**:
   - Clicca il pulsante "Start/Pause" per iniziare
   - Il pulsante diventerà rosso durante la riproduzione

### Controlli e Impostazioni

#### 🎛️ Pannello Impostazioni

- **Appearance Mode**: Seleziona tema chiaro, scuro o automatico
- **Background Image Path**: Carica un'immagine di sfondo personalizzata
- **Alpha Amount**: Controlla la trasparenza dello sfondo (0-1)
- **Noise Threshold**: Imposta la soglia minima per visualizzare il segnale
- **Frequency Bands**: Numero di barre dell'equalizzatore (1-100)
- **Number of Processors**: Configura l'elaborazione parallela

#### 🖥️ Modalità Fullscreen

- Clicca il pulsante "Fullscreen" per aprire la visualizzazione immersiva
- Usa **ESC** per uscire dalla modalità fullscreen
- Su sistemi multi-monitor, seleziona il display preferito dalle impostazioni

### ⌨️ Scorciatoie da Tastiera

- **ESC**: Esci dalla modalità fullscreen
- **Spazio**: Start/Stop cattura audio (quando l'applicazione ha il focus)

## 🔧 Configurazione Avanzata

### Ottimizzazione delle Prestazioni

#### Elaborazione Parallela
L'applicazione supporta l'elaborazione FFT parallela per migliorare le prestazioni:

- **Auto**: Calcolo automatico basato sul numero di bande
- **Manuale**: Imposta manualmente il numero di worker (1-5)

#### Raccomandazioni Hardware

| Numero di Bande | Worker Consigliati | Note |
|------------------|-------------------|------|
| 1-15 | 1 | Elaborazione single-thread |
| 16-30 | 2 | Bilanciamento CPU ottimale |
| 31-50 | 3 | Uso intensivo CPU |
| 51+ | 4-5 | Modalità fullscreen raccomandata |

### Immagini di Sfondo

Formati supportati: PNG, JPG, JPEG, GIF

Per migliori risultati:
- Risoluzione minima: 1920x1080
- Formato preferito: PNG con canale alpha
- Evita immagini troppo dettagliate che potrebbero interferire con la visualizzazione

## 📁 Struttura del Progetto

```
sound-wave/
├── main.py                          # Entry point dell'applicazione
├── requirements.txt                 # Dipendenze Python
├── resource_manager.py              # Gestione risorse (immagini, assets)
│
├── gui/                            # Interfaccia utente
│   ├── main_window_gui.py          # Finestra principale
│   ├── help_window.py              # Finestra di aiuto
│   ├── slider_frame.py             # Componente slider personalizzato
│   ├── optionmenu_frame.py         # Componente menu opzioni
│   ├── background_filepicker_frame.py # Selettore file immagini
│   └── processes_number_frame.py   # Controllo numero processori
│
├── thread/                         # Thread di elaborazione
│   ├── audioCaptureThread.py       # Cattura audio in background
│   ├── equalizer_tkinter_thread.py # Rendering equalizzatore (Tkinter)
│   ├── opengl_thread.py           # Rendering fullscreen (OpenGL)
│   └── AudioBufferAccumulator.py  # Buffer audio con overlap
│
└── resources/                      # Asset dell'applicazione
    ├── bg/                        # Immagini di sfondo
    └── icons/                     # Icone dell'interfaccia
```

## 🔬 Dettagli Tecnici

### Elaborazione Audio

- **Frequenza di Campionamento**: 44.1 kHz
- **Canali**: Stereo (2 canali)
- **Dimensione FFT**: 4096 campioni (modalità fullscreen) / 8192 (modalità anteprima)
- **Overlap**: 50% tra finestre successive
- **Finestra**: Blackman (per ridurre leakage spettrale)

### Algoritmi di Visualizzazione

#### Distribuzione Frequenze
Le bande di frequenza sono distribuite su scala logaritmica da 20 Hz a 20 kHz:

```python
def generate_frequency_bands(self, num_bands: int) -> list[tuple[float, float]]:
    """
    Genera bande di frequenza in scala logaritmica.
    
    Args:
        num_bands (int): Numero di bande desiderate
        
    Returns:
        list[tuple[float, float]]: Lista di tuple (freq_min, freq_max)
    """
```

#### Smoothing Adattivo
Il sistema implementa uno smoothing temporale adattivo che si regola automaticamente in base al refresh rate del monitor:

- **60 Hz**: Fattore base 0.25
- **120 Hz**: Fattore base 0.30  
- **144+ Hz**: Fattore base 0.35

### Rendering Graphics

#### Modalità Tkinter (Anteprima)
- Canvas-based rendering
- Frame rate: 120 FPS
- Colori: Verde (0-50%), Giallo (50-80%), Rosso (80-100%)

#### Modalità OpenGL (Fullscreen)
- Hardware-accelerated rendering
- VSync abilitato per sincronizzazione monitor
- Buffer circolari per performance ottimali
- Supporto texture per sfondi personalizzati

## 🐛 Troubleshooting

### Problemi Comuni

#### L'applicazione non rileva dispositivi audio
**Soluzione**: 
- Verifica che i driver audio siano aggiornati
- Controlla le autorizzazioni microfono (macOS/Linux)
- Riavvia l'applicazione come amministratore (Windows)

#### Performance scarse con molte bande
**Soluzioni**:
- Riduci il numero di bande di frequenza
- Usa la modalità fullscreen per performance migliori
- Aumenta il numero di worker nelle impostazioni
- Chiudi altre applicazioni CPU-intensive

#### La modalità fullscreen non si avvia
**Verifiche**:
- Controlla che i driver grafici siano aggiornati
- Verifica il supporto OpenGL del sistema
- Su Linux, assicurati che le librerie OpenGL siano installate

#### Immagine di sfondo non viene caricata
**Soluzioni**:
- Verifica che il formato immagine sia supportato (PNG, JPG, JPEG, GIF)
- Controlla i permessi di lettura del file
- Usa percorsi assoluti invece di relativi

### Log e Debug

Per abilitare il logging dettagliato, modifica le variabili in `thread/opengl_thread.py`:

```python
GENERAL_LOG = True   # Log generali
FFT_LOGS = True      # Log elaborazione FFT  
PERF_LOGS = True     # Log performance
```

## 🤝 Contributi

I contributi sono benvenuti! Per contribuire:

1. **Fork** la repository
2. Crea un **branch** per la tua feature (`git checkout -b feature/nuova-feature`)
3. **Commit** le modifiche (`git commit -am 'Aggiungi nuova feature'`)
4. **Push** sul branch (`git push origin feature/nuova-feature`)
5. Apri una **Pull Request**

### Linee Guida per i Contributi

- Segui lo stile di codifica esistente
- Aggiungi docstring per nuove funzioni/classi
- Testa le modifiche su più piattaforme quando possibile
- Aggiorna la documentazione se necessario

## 📄 Licenza

Questo progetto è rilasciato sotto licenza MIT. Vedi il file `LICENSE` per i dettagli completi.

## 🙏 Ringraziamenti

- **CustomTkinter** - Per l'interfaccia utente moderna
- **SoundCard** - Per la cattura audio cross-platform
- **PyOpenGL** - Per il rendering ad alte prestazioni
- **GLFW** - Per la gestione finestre OpenGL

---

**Sviluppato con ❤️**