# Welcome to the Grammar Scoring Engine 

A machine learning model that analyzes spoken audio to predict grammar quality using acoustic features (MFCCs, Spectral Contrast, ZCR) and provides real-time feedback through an interactive web interface.

**Audio Analysis**: Extracts 20+ acoustic features including:
  - MFCCs (Mel-frequency cepstral coefficients)
  - Spectral Contrast
  - Zero Crossing Rate (ZCR)
- **Machine Learning**: Random Forest Regressor trained on speech patterns
- **Real-time Feedback**: Instant grammar score prediction (0.0-1.0 scale)
- **Visual Analytics**:
  - Interactive waveform display
  - MFCC heatmap visualization
  - As well as Download Full csv report
- **File Support**: Processes WAV and MP3 audio formats

## Tech Stack 

**Core Technologies**
- `Librosa` (Audio feature extraction)
- `scikit-learn` (Machine Learning pipeline)
- `Streamlit` (Web interface)
- `NumPy`/`Pandas` (Data processing)

**Supporting Libraries**
- Matplotlib (Visualizations)
- Joblib (Model persistence)
- LanguageTool-Python (Grammar validation)

**Dataset 📚**
- Using Mozilla's Common Voice dataset from Kaggle:
- Annotated with speaker metadata and text transcripts
- Used a curated subset of Mozilla's Common Voice dataset:
- **Samples**: 25,403 voice recordings (`cv-invalid.csv`)
- **Features**:  
  - Audio files (MP3 format)  
  - Text transcripts  
  - Speaker metadata (age, gender, accent)
## Acknowledgments 
- Mozilla Common Voice team for speech dataset
- Librosa community for audio processing tools
- Streamlit for interactive web components


