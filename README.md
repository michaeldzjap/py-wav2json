# py-wav2json
Generate a waveform.js compatible JSON data representation of a wav file

Author: Michael Dzjaparidze

License: MIT

Inspired by [wav2json](https://github.com/beschulz/wav2json), but just a little different.

**note**: py-wav2json depends on NumPy and SciPy

# Examples
    python wav2json.py -i track.wav
will create a JSON file containing a single linearly interpolated approximation of the wav file at 800 equally spaced data points. Note that N-channel wav files are first independently interpolated before computing the average of the N channels.
