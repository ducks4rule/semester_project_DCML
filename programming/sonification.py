from midiutil import MIDIFile
from mingus.core import progressions

# transform markov output to readable chords with mignus
def chords_to_notes(progression, key='F'):
    return progressions.to_chords(progression, key)

def swap_accidentals(note):
    if note == 'Db':
        return 'C#'
    if note == 'D#':
        return 'Eb'
    if note == 'E#':
        return 'F'
    if note == 'Gb':
        return 'F#'
    if note == 'G#':
        return 'Ab'
    if note == 'A#':
        return 'Bb'
    if note == 'B#':
        return 'C'
    if note == 'Cb':
        return 'B'
    if note == 'Bbb':
        return 'A'
    return note

def note_to_number(note: str, octave: int) -> int:
    note = swap_accidentals(note)
    NOTES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    OCTAVES = list(range(11))
    NOTES_IN_OCTAVE = len(NOTES)
    note = swap_accidentals(note)
    if note not in NOTES:
        print('error: ', note)
    assert note in NOTES
    assert octave in OCTAVES

    note = NOTES.index(note)
    note += (NOTES_IN_OCTAVE * octave)
    return note

def midi_to_wav(midi_file):
    soundfont = '~/Documents/Musik/Soundfont/GeneralUser GS 1.471/GeneralUser GS v1.471.sf2'
    # convert midi to wav using fluidsynth
    wav_file = midi_file.replace('.mid', '.wav')
    os.system(f'fluidsynth -ni {soundfont} {midi_file} -F {wav_file} -r 44100')
    os.remove(midi_file)

# create a midi file
def create_midi_file(chord_progression, filename='output.mid',
                     duration=2, vol=100, octave=4):
    # create a midi file
    midi = MIDIFile(1)
    midi.addTempo(0, 0, 120)

    # change chord symbols to notes
    note_progression = chords_to_notes(chord_progression)
    print(type(note_progression))

    # add chords
    time = 0
    for chord in note_progression:
        for pitch in chord:
            pitch = note_to_number(pitch, octave)
            midi.addNote(0, 0, pitch, time, duration , vol)
        time += duration

    # write to file
    with open(filename, 'wb') as f:
        midi.writeFile(f)

    # convert to wav
    midi_to_wav(filename)
    print('midi file ' + filename + ' created')

# ==========================================
# test, from notebook
# ==========================================

# start = 'I'
# length = 32
# dur = 2
# chords = markov.predict(start, n=length, start_at_current=True)
#
# create_midi_file(chords, filename='markov_chain_output.mid', duration=dur)
#
# org_chords = df.values[:length]
# create_midi_file(org_chords, filename='original_output.mid', duration=dur)
#
