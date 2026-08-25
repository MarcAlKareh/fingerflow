import music21 as m21
from pathlib import Path
from typing import Dict, List, Any

def extract_hand_data(part: m21.stream.Part, start_id: int = 0) -> List[Dict[str, Any]]:
    """Extracts note events from a single staff/part."""
    hand_data = []
    note_id = start_id
    
    # .flat flattens the hierarchy into a single linear timeline
    notes_to_parse = part.flat.notes
    
    for element in notes_to_parse:
        # Calculate time in seconds (assuming default 120 BPM if no tempo track is found)
        start_time_sec = float(element.offset) * 0.5
        duration_sec = float(element.quarterLength) * 0.5
        
        if isinstance(element, m21.note.Note):
            hand_data.append({
                "note_id": note_id,
                "pitch": element.pitch.midi,
                "start_time_sec": start_time_sec,
                "duration_sec": duration_sec
            })
            note_id += 1
        elif isinstance(element, m21.chord.Chord):
            # For chords, we add each note sharing the same start time
            for pitch in element.pitches:
                hand_data.append({
                    "note_id": note_id,
                    "pitch": pitch.midi,
                    "start_time_sec": start_time_sec,
                    "duration_sec": duration_sec
                })
                note_id += 1
                
    return hand_data

def parse_musicxml(mxl_path: Path) -> Dict[str, Any]:
    """Ingests a MusicXML file and outputs a structured dictionary tree."""
    if not mxl_path.exists():
        raise FileNotFoundError(f"Cannot find MusicXML file at {mxl_path}")
        
    try:
        score = m21.converter.parse(mxl_path)
    except Exception as e:
        raise RuntimeError(f"Failed to parse MusicXML: {e}") from e

    parts = score.parts
    
    # Standard Piano grand staff: Part 0 is Right Hand, Part 1 is Left Hand
    right_hand_stream = parts[0] if len(parts) > 0 else m21.stream.Part()
    left_hand_stream = parts[1] if len(parts) > 1 else m21.stream.Part()
    
    tempo_bpm = 120  # Fallback
    metronome_marks = score.flat.getElementsByClass(m21.tempo.MetronomeMark)
    if metronome_marks:
        tempo_bpm = metronome_marks[0].number
        
    return {
        "tempo_bpm": tempo_bpm,
        "right_hand": extract_hand_data(right_hand_stream, start_id=0),
        "left_hand": extract_hand_data(left_hand_stream, start_id=1000)
    }