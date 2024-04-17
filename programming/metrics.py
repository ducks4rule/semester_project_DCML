import numpy as np
import pandas as pd

import chord_eval as ce



def tone_by_tone_distance(
    root1: int,
    root2: int,
    chord_type1: ChordType,
    chord_type2: ChordType,
    inversion1: int = 0,
    inversion2: int = 0,
    changes1: str = None,
    changes2: str = None,
    root_bonus: int = 1,
    bass_bonus: int = 1,
    pitch_type: PitchType = PitchType.MIDI,
) -> float:
    """
    From https://github.com/DCMLab/chord-eval/blob/19ea2b74a9b66ae9a4c3f64a0a7564be9ee9d90c/chord_eval/metric.py
    Adapted

    Get the tone by tone distance between two chords: The average of the proporion
    of pitch classes in each chord that are unmatched in the other chord.

    Parameters
    ----------
    root1 : int
        The root of the given first chord, either as a MIDI note number or as
        a TPC note number (interval above C in fifths with F = -1, C = 0, G = 1, etc.).
        If the chord is some inversion, the root pitch will be on this pitch, but there
        may be other pitches below it.

    root2 : int
        The root of the 2nd chord.

    chord_type1 : ChordType
        The chord type of the given first chord.

    chord_type2 : ChordType
        The chord type of the given second chord.

    inversion1 : int, optional
        The inversion of the first chord.
        The default is 0.

    inversion2 : int, optional
        The inversion of the second chord.
        The default is 0.

    changes1 : str
        Any alterations to the 1st chord's pitches, as a semi-colon separated string.
        Each alteration should be in the form "orig:new", where "orig" represents
        the original pitch that has been altered to "new". "orig" can also be blank
        for added pitches, and "new" can be prepended with "+" to indicate that a
        pitch occurs in an upper octave (e.g., a C7 chord with a 9th is represented
        by ":+1", using MIDI pitch). Note that TPC pitch does not allow for the
        representation of different octaves so any "+" is ignored.

    changes2 : str
        Any alterations to the 2nd chord's pitches.

    root_bonus : int, optional
        Give additional accuracy to chords whose root notes match (and penalize those whose
        do not). 1 will give an additional bonus or penalty of equal weighting as each other
        note.

        For example, for a C major vs C minor comparison:
            - Default distance (root_bonus == 0): 1/3 (each chord matches 2/3 of their notes
              with the other chord).
            - With root_bonus == 1: 1/4 (each chord matches 2/3 of their notes with the other
              chord, plus an additional 1 "bonus match" up to 3/4 for their roots matching).

        For example, for C major vs Cmin7 comparison:
            - Default distance: 5 / 12 (0.417 -- the average of 1/3 from the C major triad and
              2/4 from the Cmin7).
            - With root_bonus == 1: 13/40 (0.325 -- the average of 1/4 from the C major triad
              and 2/5 from the Cmin7).

        For example, for C major vs A minor:
            - Default distance: 1/3
            - With root_bonus == 1: 1/2

    bass_bonus : int, optional
        Give additional accuracy to chords whose bass notes match. 1 will give an additional
        bonus of equal weighting as each other note.

        For example, for a C major vs C minor comparison:
            - Default distance (bass_bonus == 0): 1/3 (each chord matches 2/3 of their notes
              with the other chord).
            - With bass_bonus == 1: 1/4 (each chord matches 2/3 of their notes with the other
              chord, plus an additional 1 "bonus match" up to 3/4 for their bass notes matching).

        For example, for C major vs Cmin7 comparison:
            - Default distance: 5 / 12 (0.417 -- the average of 1/3 from the C major triad and
              2/4 from the Cmin7).
            - With bass_bonus == 1: 13/40 (0.325 -- the average of 1/4 from the C major triad
              and 2/5 from the Cmin7).

        For example, for C major vs A minor:
            - Default distance: 1/3
            - With bass_bonus == 1: 1/2

    pitch_type : PitchType
        The pitch type in which root notes are encoded, and using which tone-by-tone distance
        should be calculated. With MIDI (default), enharmonic equivalence is assumed. Otherwise,
        C# and Db count as different pitches.

    Returns
    -------
    distance : float
        The tone by tone distance between two chords.
    """

    def one_sided_tbt(
        note_set1: set,
        note_set2: set,
        root_matches: bool,
        bass_matches: bool,
        root_bonus: int,
        bass_bonus: int,
    ) -> float:
        """
        Get the one-sided tbt. That is, the proportion of chord 1's notes which are missing
        from chord2, including root and bass bonus.

        Parameters
        ----------
        note_set1 : set
            The set of pitch classes contained in chord 1.
        note_set2 : np.ndarray
            The set of pitch classes in chord 2.
        root_matches : bool
            True if the chord roots match. False otherwise.
        bass_matches : bool
            True if the chord bass notes match. False otherwise.
        root_bonus : int
            The root bonus to use (see the tone_by_tone_distance function's comments for details).
        bass_bonus : int
            The bass bonus to use (see the tone_by_tone_distance function's comments for details).

        Returns
        -------
        distance : float
            The one-sided tone-by-tone distance.
        """
        matches = len(note_set1.intersection(note_set2))

        if root_matches:
            matches += root_bonus
        if bass_matches:
            matches += bass_bonus

        return 1 - matches / (len(note_set1) + bass_bonus + root_bonus)

    notes1 = get_chord_pitches(
        root=root1,
        chord_type=chord_type1,
        pitch_type=PitchType.MIDI,
        inversion=inversion1,
        changes=changes1,
    )
    if pitch_type == PitchType.MIDI:
        notes1 %= 12

    notes2 = get_chord_pitches(
        root=root2,
        chord_type=chord_type2,
        pitch_type=PitchType.MIDI,
        inversion=inversion2,
        changes=changes2,
    )
    if pitch_type == PitchType.MIDI:
        notes2 %= 12

    root_matches = root1 == root2
    bass_matches = notes1[0] == notes2[0]

    distance = np.mean(
        [
            one_sided_tbt(
                set(notes1),
                set(notes2),
                root_matches,
                bass_matches,
                root_bonus,
                bass_bonus,
            ),
            one_sided_tbt(
                set(notes2),
                set(notes1),
                root_matches,
                bass_matches,
                root_bonus,
                bass_bonus,
            ),
        ]
    )

    return distance
