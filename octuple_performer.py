"""Modified version of Octuple with no Program (Track) tokens
To use mainly for tasks handling a single track.

"""

from math import ceil
import json, collections
from pathlib import Path, PurePath
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
from miditoolkit import MidiFile, Instrument, Note, TempoChange, ControlChange, TimeSignature

from miditok.midi_tokenizer_base import MIDITokenizer
from miditok.vocabulary import Vocabulary, Event
from miditok.constants import *
from miditok.utils import remove_duplicated_notes
from decimal import Decimal
import math, copy


DEFAULT_VELOCITY_BINS = np.linspace(10, 128, 8, dtype=np.int32)
MIN_VELOCITY = 10
DEFAULT_RESOLUTION = 384

# MatchTuple -------------------------
BestQuantizationMatch = collections.namedtuple('BestQuantizationMatch',
    ['error', 'tick', 'match', 'signedError', 'divisor'])

class OctuplePerformer(MIDITokenizer):
    r"""Modified version of Octuple with no Program (Track) tokens
    To use mainly for tasks handling a single track.

    :param pitch_range: range of used MIDI pitches
    :param beat_res: beat resolutions, with the form:
            {(beat_x1, beat_x2): beat_res_1, (beat_x2, beat_x3): beat_res_2, ...}
            The keys of the dict are tuples indicating a range of beats, ex 0 to 3 for the first bar
            The values are the resolution, in samples per beat, of the given range, ex 8
    :param nb_velocities: number of velocity bins
    :param additional_tokens: specifies additional tokens (time signature, tempo)
    :param sos_eos_tokens: adds Start Of Sequence (SOS) and End Of Sequence (EOS) tokens to the vocabulary
    :param mask: will add a MASK token to the vocabulary (default: False)
    :param params: can be a path to the parameter (json encoded) file or a dictionary
    """
    
    @staticmethod
    def nearestMultiple(n, unit):
        if n < 0:
            raise ValueError(f'n ({n}) is less than zero. '
                            + 'Thus cannot find nearest multiple for a value '
                            + f'less than the unit, {unit}')
        
        n = Decimal(str(n))
        unit = Decimal(str(unit))

        mult = math.floor(n / unit)  # can start with the floor
        mult = Decimal(str(mult))
        halfUnit = unit / Decimal('2.0')
        halfUnit = Decimal(str(halfUnit))


        matchLow = unit * mult
        matchHigh = unit * (mult + 1)

        # print(['mult, halfUnit, matchLow, matchHigh', mult, halfUnit, matchLow, matchHigh])

        if matchLow >= n >= matchHigh:
            raise Exception(f'cannot place n between multiples: {matchLow}, {matchHigh}')

        if matchLow <= n <= (matchLow + halfUnit):
            return float(matchLow), float(round(n - matchLow, 7)), float(round(n - matchLow, 7))
        else:
            return float(matchHigh), float(round(matchHigh - n, 7)), float(round(n - matchHigh, 7))
    
    def __init__(self, 
                 pitch_range: range = PITCH_RANGE,
                 beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
                 nb_velocities: int = NB_VELOCITIES, 
                 additional_tokens: Dict[str, bool] = ADDITIONAL_TOKENS,
                 sos_eos_tokens: bool = False, 
                 mask: bool = False, 
                 num_of_performer: int = 49, 
                 num_of_composition: int=1562, 
                 is_quantize: str='None', 
                 params=None):
        additional_tokens['Chord'] = False  # Incompatible additional token
        additional_tokens['Rest'] = False
        additional_tokens['Program'] = False
        # used in place of positional encoding
        self.programs = list(range(-1, 128))
        self.max_bar_embedding = 60  # this attribute might increase during encoding
        self.num_of_performer = num_of_performer
        self.num_of_composition = num_of_composition
        self.is_quantize = is_quantize
        super().__init__(pitch_range, beat_res, nb_velocities, additional_tokens, sos_eos_tokens, mask, params)

    def save_params(self, out_dir: Union[str, Path, PurePath]):
        r"""Override the parent class method to include additional parameter drum pitch range
        Saves the base parameters of this encoding in a txt file
        Useful to keep track of how a dataset has been tokenized / encoded
        It will also save the name of the class used, i.e. the encoding strategy

        :param out_dir: output directory to save the file
        """
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(PurePath(out_dir, 'config').with_suffix(".txt"), 'w') as outfile:
            json.dump({'pitch_range': (self.pitch_range.start, self.pitch_range.stop),
                       'beat_res': {f'{k1}_{k2}': v for (k1, k2), v in self.beat_res.items()},
                       'nb_velocities': len(self.velocities),
                       'additional_tokens': self.additional_tokens,
                       '_sos_eos': self._sos_eos,
                       '_mask': self._mask,
                       'encoding': self.__class__.__name__,
                       'max_bar_embedding': self.max_bar_embedding},
                      outfile)
    
    def match_tempo(self, note):
        tempo_changes = np.asarray([[i.time, i.tempo] for i in self.current_midi_metadata['tempo_changes']])
        index = np.argmin(abs(note.start-tempo_changes[:,0]))
        return tempo_changes[index][1]

    def time_quantize_by_group(self, notes):
        min_interval = DEFAULT_RESOLUTION / 60000 * 25
        group = []
        note_index = []
        onset = 0
        for i, note in enumerate(notes):
            if note.pitch > self.pitch_range.stop:
                continue
            if group == []:
                group.append(note.start)
                note_index.append(i)
                onset = note.start
            elif note.start - onset < (min_interval * self.match_tempo(note)):
                group.append(note.start)
                note_index.append(i)
                onset = note.start
            elif note.start - onset >= (min_interval * self.match_tempo(note)):
                try:
                    mean_onset = int(np.round(np.mean(group)))
                except ValueError:
                    print(group)
                for j in note_index:
                    offset = mean_onset - notes[j].start
                    notes[j].start = mean_onset
                    notes[j].end += offset
                group = [note.start]
                note_index = [i]
                onset = note.start
        return notes
    
    def time_quantize_by_grid(self, note, quarterLengthDivisors=[32,24]):
        # this presently is not trying to avoid overlaps that
        # result from quantization; this may be necessary

        def bestMatch(target, divisors):
            found = []
            for div in divisors:
                match, error, signedErrorInner = self.nearestMultiple(target, (self.current_midi_metadata['time_division']/div))
                # Sort by unsigned error, then "tick" (divisor expressed as QL, e.g. 0.25)
                found.append(BestQuantizationMatch(error, self.current_midi_metadata['time_division']/div, match, signedErrorInner, div))
            # get first, and leave out the error
            bestMatchTuple = sorted(found)[0]
            return bestMatchTuple
        
        e = note.end
        sign = 1
        if e < 0:
            sign = -1                                                                                  
            e = -1 * e
        e_matchTuple = bestMatch(float(e), quarterLengthDivisors)
        note.end = e_matchTuple.match * sign
        
        s = note.start
        sign = 1
        if s < 0:
            sign = -1
            s = -1 * s
        s_matchTuple = bestMatch(float(s), quarterLengthDivisors)
        note.start = s_matchTuple.match * sign         
        
        # diff = note.start - s
        # note.end += diff   
        
        if note.start == note.end:
            note.end += int(self.current_midi_metadata['time_division'] / max(self.beat_res.values()))
            
        return note
            
    def velocity_quantize(self, note):
        vel = note.velocity
        if vel == 0:
            return vel
        vel_q = DEFAULT_VELOCITY_BINS[
            np.argmin(abs(DEFAULT_VELOCITY_BINS-vel))]
        vel_q = max(MIN_VELOCITY, vel_q)
        vel_q = int(np.round(vel_q))
        vel_q = self.velocities[int(np.argmin(np.abs(self.velocities - vel_q)))]
        return vel_q
    
    def midi_to_tokens(self, midi: MidiFile, performer: int = 0, composition: int = 0, *args, **kwargs) -> List[List[Union[int, List[int]]]]:
        r"""Converts a MIDI file in a tokens representation.
        NOTE: if you override this method, be sure to keep the first lines in your method

        :param midi: the MIDI objet to convert
        :return: the token representation, i.e. tracks converted into sequences of tokens
        """
        # Check if the durations values have been calculated before for this time division
        if midi.ticks_per_beat not in self.durations_ticks:
            self.durations_ticks[midi.ticks_per_beat] = np.array([(beat * res + pos) * midi.ticks_per_beat // res
                                                                  for beat, pos, res in self.durations])
            
        # Preprocess the MIDI file
        self.preprocess_midi(midi)
        # Register MIDI metadata
        self.current_midi_metadata = {'time_division': midi.ticks_per_beat,
                                      'tempo_changes': midi.tempo_changes,
                                      'time_sig_changes': midi.time_signature_changes,
                                      'key_sig_changes': midi.key_signature_changes}

        # **************** OVERRIDE FROM HERE, KEEP THE LINES ABOVE IN YOUR METHOD ****************
        # Check bar embedding limit, update if needed
        
        control_change_times = []
        for t in range(len(midi.instruments)):
            ticks = [i.time for i in midi.instruments[t].control_changes]
            control_change_times += ticks
        control_change_times.append(midi.max_tick)
        midi.max_tick = max(control_change_times)
        
        nb_bars = ceil(midi.max_tick / (midi.ticks_per_beat * 4))
        if self.max_bar_embedding < nb_bars:
            self.vocab[4].add_event(f'Bar_{i}' for i in range(self.max_bar_embedding, nb_bars))
            self.max_bar_embedding = nb_bars    

        tokens = []
        if self.is_quantize:
            q_tokens = []
            for track in midi.instruments:
                if track.program in self.programs:
                    token, q_token= self.track_to_tokens(track, performer, composition)
                    tokens += token
                    q_tokens += q_token
                    
            tokens, q_tokens = zip(*sorted(zip(tokens, q_tokens), key=lambda x: (x[0][0].time, x[0][0].desc, x[0][0].value)))
            # Convert pitch events into tokens
            for time_step in tokens:
                time_step[0] = self.vocab[0].event_to_token[f'{time_step[0].type}_{time_step[0].value}']
                
            for time_step in q_tokens:
                time_step[0] = self.vocab[0].event_to_token[f'{time_step[0].type}_{time_step[0].value}']
                
            return tokens, q_tokens
       
        else:
            for track in midi.instruments:
                if track.program in self.programs:
                    tokens += self.track_to_tokens(track, performer, composition)
            tokens.sort(key=lambda x: (x[0].time, x[0].desc, x[0].value))  # Sort by time then track then pitch
            
            # Convert pitch events into tokens
            for time_step in tokens:
                time_step[0] = self.vocab[0].event_to_token[f'{time_step[0].type}_{time_step[0].value}']

            return tokens

    def track_to_tokens(self, track: Instrument, performer: int = 0, composition: int=0) -> List[List[int]]:
        r"""Converts a track (miditoolkit.Instrument object) into a sequence of tokens
        A time step is a list of tokens where:
            (list index: token type)
            0: Pitch
            1: Velocity
            2: Duration
            3: Position
            4: Bar
            5: Performer
            (6: Tempo)
            (7: Composition)

        :param track: MIDI track to convert
        :return: sequence of corresponding tokens
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        ticks_per_sample = self.current_midi_metadata['time_division'] / max(self.beat_res.values())
        ticks_per_bar = self.current_midi_metadata['time_division'] * 4
        dur_bins = self.durations_ticks[self.current_midi_metadata['time_division']]
        
        if self.additional_tokens['Pedal']:
            control_changes = []
            track.control_changes.sort(key=lambda x: (x.time))
            for idx, change in enumerate(track.control_changes):
                for i in range(idx+1, len(track.control_changes)):
                    if (track.control_changes[i].number == change.number):
                        end = track.control_changes[i].time
                        break
                change.value = self.velocities[int(np.argmin(np.abs(self.velocities - change.value)))] if change.value != 0 else 0
                control_changes.append(Note(change.value, change.number-64+self.pitch_range.stop, change.time, end))
        
        events = []
        current_tick = -1
        current_bar = -1
        current_pos = -1
        
        current_tick_q = -1
        current_bar_q = -1
        current_pos_q = -1
        
        current_tempo_idx = 0
        current_tempo = self.current_midi_metadata['tempo_changes'][current_tempo_idx].tempo
        tempo_mean = int(np.ceil(np.mean([x.tempo for x in self.current_midi_metadata['tempo_changes']])))
        
        if self.is_quantize != 'None':
            events_q = []
        
        if self.additional_tokens['Pedal']:
            track.notes  += control_changes
            track.notes.sort(key=lambda x: (x.start))
        
        for note in track.notes:
            # if (note.pitch < self.pitch_range.stop) or (current_pos == -1):
            #     # Positions and bars
            if note.start != current_tick:
                pos_index = int((note.start % ticks_per_bar) / ticks_per_sample)
                current_tick = note.start
                current_bar = current_tick // ticks_per_bar
                current_pos = pos_index

            # Note attributes
            duration = note.end - note.start
            dur_index = np.argmin(np.abs(dur_bins - duration))
            event = [Event(type_='Pitch', time=note.start, value=note.pitch,
                        desc=-1 if track.is_drum else track.program),
                        self.vocab[1].event_to_token[f'Velocity_{note.velocity}'],
                        self.vocab[2].event_to_token[f'Duration_{".".join(map(str, self.durations[dur_index]))}'],
                        self.vocab[3].event_to_token[f'Position_{current_pos}'],
                        self.vocab[4].event_to_token[f'Bar_{current_bar}'],
                        self.vocab[5].event_to_token[f'Performer_{performer}'],
                        ]

            if self.is_quantize == "Grid":
                q_note = copy.deepcopy(note)
                q_note = self.time_quantize_by_grid(q_note)
                
                if q_note.start != current_tick_q:
                    pos_index = int((q_note.start % ticks_per_bar) / ticks_per_sample)
                    current_tick_q = q_note.start
                    current_bar_q = int(current_tick_q // ticks_per_bar)
                    current_pos_q = pos_index
                
                
                duration = q_note.end - q_note.start
                dur_index = np.argmin(np.abs(dur_bins - duration))
                q_vel = self.velocity_quantize(q_note)
                event_q = [Event(type_='Pitch', time=q_note.start, value=q_note.pitch,
                            desc=-1 if track.is_drum else track.program),
                            self.vocab[1].event_to_token[f'Velocity_{q_vel}'],
                            self.vocab[2].event_to_token[f'Duration_{".".join(map(str, self.durations[dur_index]))}'],
                            self.vocab[3].event_to_token[f'Position_{current_pos_q}'],
                            self.vocab[4].event_to_token[f'Bar_{current_bar_q}'],
                            self.vocab[5].event_to_token[f'Performer_{performer}'],
                            ]

            # (Tempo)
            if self.additional_tokens['Tempo']:
                # If the current tempo is not the last one
                if current_tempo_idx + 1 < len(self.current_midi_metadata['tempo_changes']):
                    # Will loop over incoming tempo changes
                    for tempo_change in self.current_midi_metadata['tempo_changes'][current_tempo_idx + 1:]:
                        # If this tempo change happened before the current moment
                        if tempo_change.time <= note.start:
                            current_tempo = tempo_change.tempo
                            current_tempo_idx += 1  # update tempo value (might not change) and index
                        elif tempo_change.time > note.start:
                            break  # this tempo change is beyond the current time step, we break the loop
                event.append(self.vocab[6].event_to_token[f'Tempo_{current_tempo}'])
                if self.is_quantize == "Grid":
                    event_q.append(self.vocab[6].event_to_token[f'Tempo_{tempo_mean}'])
                    if self.additional_tokens['Composition']:
                        event_q.append(self.vocab[7].event_to_token[f'Composition_{composition}'])

            if self.additional_tokens['Composition']:
                event.append(self.vocab[7].event_to_token[f'Composition_{composition}'])

            events.append(event)
            
            if self.is_quantize == "Grid":
                events_q.append(event_q)

        if self.is_quantize == "Group":
            current_tick = -1
            current_bar = -1
            current_pos = -1
            
            current_tick_q = -1
            current_bar_q = -1
            current_pos_q = -1
            
            current_tempo_idx = 0
            current_tempo = self.current_midi_metadata['tempo_changes'][current_tempo_idx].tempo
            quantized_notes = self.time_quantize_by_group(track.notes)
            
            for note in quantized_notes:
                # note = self.time_quantize_by_grid(note)
                # if (note.pitch < self.pitch_range.stop) or (current_pos == -1):
                    # Positions and bars
                if note.start != current_tick:
                    pos_index = int((note.start % ticks_per_bar) / ticks_per_sample)
                    current_tick = note.start
                    current_bar = int(current_tick // ticks_per_bar)
                    current_pos = pos_index

                # Note attributes
                duration = note.end - note.start
                dur_index = np.argmin(np.abs(dur_bins - duration))
                q_vel = self.velocity_quantize(note)
                event_q = [Event(type_='Pitch', time=note.start, value=note.pitch,
                            desc=-1 if track.is_drum else track.program),
                            self.vocab[1].event_to_token[f'Velocity_{q_vel}'],
                            self.vocab[2].event_to_token[f'Duration_{".".join(map(str, self.durations[dur_index]))}'],
                            self.vocab[3].event_to_token[f'Position_{current_pos}'],
                            self.vocab[4].event_to_token[f'Bar_{current_bar}'],
                            self.vocab[5].event_to_token[f'Performer_{performer}'],
                            ]
                # (Tempo)
                if self.additional_tokens['Tempo']:
                    event_q.append(self.vocab[6].event_to_token[f'Tempo_{tempo_mean}'])
                
                if self.additional_tokens['Composition']:
                    event_q.append(self.vocab[7].event_to_token[f'Composition_{composition}'])

                events_q.append(event_q)    
        
        if self.is_quantize != None:
            return events, events_q
        else:
            return events

    def tokens_to_track(self, tokens: List[List[int]], time_division: Optional[int] = TIME_DIVISION,
                        program: Optional[Tuple[int, bool]] = (0, False)) -> Tuple[Instrument, List[TempoChange]]:
        r"""Converts a sequence of tokens into a track object
        A time step is a list of tokens where:
            (list index: token type)
            0: Pitch
            1: Velocity
            2: Duration
            3: Position
            4: Bar
            5: Performer
            (6: Tempo)

        :param tokens: sequence of tokens to convert
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :return: the miditoolkit instrument object and tempo changes
        """
        assert time_division % max(self.beat_res.values()) == 0, \
            f'Invalid time division, please give one divisible by {max(self.beat_res.values())}'
        events = self.tokens_to_events(tokens, multi_voc=True)
        
        ticks_per_sample = time_division // max(self.beat_res.values())
        name = 'Drums' if program[1] else MIDI_INSTRUMENTS[program[0]]['name']
        instrument = Instrument(program[0], is_drum=program[1], name=name)

        tempo_changes = [TempoChange(TEMPO, 0)]
        if self.additional_tokens['Tempo']:
            for i in range(len(events)):
                if events[i][-1].value != 'None':
                    tempo_changes = [TempoChange(int(events[i][-1].value), 0)]
                    break

        for time_step in events:
            if any(tok.value == 'None' for tok in time_step[:6]):
                continue  # Either padding, mask: error of prediction or end of sequence anyway

            # Note attributes
            pitch = int(time_step[0].value)
            vel = int(time_step[1].value)
            duration = self._token_duration_to_ticks(time_step[2].value, time_division)

            # Time and track values
            current_pos = int(time_step[3].value)
            current_bar = int(time_step[4].value)
            current_tick = current_bar * time_division * 4 + current_pos * ticks_per_sample

            # Append the created note
            if pitch < self.pitch_range.stop:
                instrument.notes.append(Note(vel, pitch, current_tick, current_tick + duration))
            else:
                instrument.control_changes.append(ControlChange(pitch + 64 - self.pitch_range.stop, vel, current_tick))
                

            # Tempo, adds a TempoChange if necessary
            if self.additional_tokens['Tempo'] and time_step[-1].value != 'None':
                tempo = int(time_step[-1].value)
                if tempo != tempo_changes[-1].tempo:
                    tempo_changes.append(TempoChange(tempo, current_tick))
            
        # print(instrument.control_changes)
        return instrument, tempo_changes

    def _create_vocabulary(self, sos_eos_tokens: bool = None) -> List[Vocabulary]:
        r"""Creates the Vocabulary object of the tokenizer.
        See the docstring of the Vocabulary class for more details about how to use it.
        NOTE: token index 0 is often used as a padding index during training

        :param sos_eos_tokens: DEPRECIATED, will include Start Of Sequence (SOS) and End Of Sequence (tokens)
        :return: the vocabulary object
        """
        if sos_eos_tokens is not None:
            print(f'\033[93msos_eos_tokens argument is depreciated and will be removed in a future update, '
                  f'_create_vocabulary now uses self._sos_eos attribute set a class init \033[0m')
        vocab = [Vocabulary({'PAD_None': 0}, sos_eos=self._sos_eos, mask=self._mask) for _ in range(6)]

        # PITCH
        vocab[0].add_event(f'Pitch_{i}' for i in self.pitch_range)
        if self.additional_tokens['Pedal']:
            vocab[0].add_event(f'Pitch_{i}' for i in range(self.pitch_range.stop, self.pitch_range.stop+4))

        # VELOCITY
        vocab[1].add_event(f'Velocity_{i}' for i in self.velocities)
        vocab[1].add_event(f'Velocity_{0}')

        # DURATION
        vocab[2].add_event(f'Duration_{".".join(map(str, duration))}' for duration in self.durations)

        # POSITION
        nb_positions = max(self.beat_res.values()) * 4  # 4/4 time signature
        vocab[3].add_event(f'Position_{i}' for i in range(nb_positions))

        # BAR
        vocab[4].add_event(f'Bar_{i}' for i in range(self.max_bar_embedding))  # bar embeddings (positional encoding)
        
        # Performer
        vocab[5].add_event(f'Performer_{i}' for i in range(self.num_of_performer))

        # TEMPO
        if self.additional_tokens['Tempo']:
            vocab.append(Vocabulary({'PAD_None': 0}, sos_eos=self._sos_eos, mask=self._mask))
            vocab[6].add_event(f'Tempo_{i}' for i in range(1, max(self.tempos)+1))
            
        # Composition
        if self.additional_tokens['Composition']:
            vocab.append(Vocabulary({'PAD_None': 0}, sos_eos=self._sos_eos, mask=self._mask))
            vocab[7].add_event(f'Composition_{i}' for i in range(self.num_of_composition))

        return vocab

    def _create_token_types_graph(self) -> Dict[str, List[str]]:
        r"""Returns a graph (as a dictionary) of the possible token
        types successions.
        Not relevant for Octuple.

        :return: the token types transitions dictionary
        """
        return {}  # not relevant for this encoding

    def token_types_errors(self, tokens: List[List[int]], consider_pad: bool = False) -> float:
        r"""Checks if a sequence of tokens is constituted of good token values and
        returns the error ratio (lower is better).
        The token types are always the same in Octuple so this methods only checks
        if their values are correct:
            - a bar token value cannot be < to the current bar (it would go back in time)
            - same for positions
            - a pitch token should not be present if the same pitch is already played at the current position

        :param tokens: sequence of tokens to check
        :param consider_pad: if True will continue the error detection after the first PAD token (default: False)
        :return: the error ratio (lower is better)
        """
        err = 0
        current_bar = current_pos = -1
        current_pitches = []

        for token in tokens:
            if consider_pad and all(token[i] == self.vocab[i]['PAD_None'] for i in range(len(token))):
                break
            if any(self.vocab[i][token].split('_')[1] == 'None' for i, token in enumerate(token)):
                err += 1
                continue
            has_error = False
            bar_value = int(self.vocab[4].token_to_event[token[4]].split('_')[1])
            pos_value = int(self.vocab[3].token_to_event[token[3]].split('_')[1])
            pitch_value = int(self.vocab[0].token_to_event[token[0]].split('_')[1])

            # Bar
            if bar_value < current_bar:
                has_error = True
            elif bar_value > current_bar:
                current_bar = bar_value
                current_pos = -1
                current_pitches = []

            # Position
            if pos_value < current_pos:
                has_error = True
            elif pos_value > current_pos:
                current_pos = pos_value
                current_pitches = []

            # Pitch
            if pitch_value in current_pitches:
                has_error = True
            else:
                current_pitches.append(pitch_value)

            if has_error:
                err += 1

        return err / len(tokens)
