import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import collections
import pretty_midi
import tensorflow as tf
import os
import glob
from scipy.io import wavfile
from midi2audio import FluidSynth

BASE_PATH = '.'
def getFile(x):
  path = os.path.join(BASE_PATH,x)
  if os.path.exists(path):
    return path
  raise FileNotFoundError

def midi_to_notes(midi_file: str) -> pd.DataFrame:
  pm = pretty_midi.PrettyMIDI(midi_file)
  instrument = pm.instruments[0]
  notes = collections.defaultdict(list) #ตั้งค่า default ให้ dictionary

  # Sort the notes by start time เรียง note ตามเวลา
  sorted_notes = sorted(instrument.notes, key=lambda note: note.start) #เป็นฟังก์ชั่นที่ใช้แล้ว list ก็จะเหมือนเดิม ค่าของตัวแปรไม่มีการเปลี่ยนแปลง
  prev_start = sorted_notes[0].start

  for note in sorted_notes:
    start = note.start
    end = note.end
    notes['pitch'].append(note.pitch)
    notes['start'].append(start)
    notes['end'].append(end)
    notes['step'].append(start - prev_start)
    notes['duration'].append(end - start)
    prev_start = start

  return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
  mse = (y_true - y_pred) ** 2
  positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
  return tf.reduce_mean(mse + positive_pressure)

def predict_next_note(
    notes: np.ndarray, 
    keras_model: tf.keras.Model, 
    temperature: float = 1.0) -> int:

  assert temperature > 0

  # Add batch dimension
  inputs = tf.expand_dims(notes, 0)

  predictions = keras_model.predict(inputs)
  pitch_logits = predictions['pitch']
  step = predictions['step']
  duration = predictions['duration']

  pitch_logits /= temperature
  pitch = tf.random.categorical(pitch_logits, num_samples=1)
  pitch = tf.squeeze(pitch, axis=-1)
  duration = tf.squeeze(duration, axis=-1)
  step = tf.squeeze(step, axis=-1)

  # `step` and `duration` values should be non-negative
  step = tf.maximum(0, step)
  duration = tf.maximum(0, duration)

  return int(pitch), float(step), float(duration)

def notes_to_midi(notes: pd.DataFrame,out_file: str, instrument_name: str,velocity: int = 100,  #note loudness
                  ) -> pretty_midi.PrettyMIDI:

  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(
      program=pretty_midi.instrument_name_to_program(
          instrument_name))

  prev_start = 0
  for i, note in notes.iterrows():
    start = float(prev_start + note['step'])
    end = float(start + note['duration'])
    note = pretty_midi.Note(
        velocity=velocity,
        pitch=int(note['pitch']),
        start=start,
        end=end,
    )
    instrument.notes.append(note)
    prev_start = start

  pm.instruments.append(instrument)
  pm.write(out_file)
  return pm

def display_audio(pm: pretty_midi.PrettyMIDI):
  waveform = pm.fluidsynth(fs=_SAMPLING_RATE)
  return waveform

_SAMPLING_RATE = 16000
partDataset = 'Dataset/'
partModel = 'Model/'

st.title('🎼 Classical Music Generator 🎹')

file_names = [fn for fn in os.listdir(partDataset)]
file_names.sort()
Original_song = st.selectbox('Please select classical music', file_names, key='1')

model_names = [fn for fn in os.listdir(partModel)]
model_names.sort()
model_choice = st.selectbox('Select your desired artist', model_names, key='1')

num_predictions = st.slider( "Sequence Lenght of Note", min_value=100 , max_value=500 ,value=300, step=1)

submit = st.button('Generate Classical Music')

seq_length = 50 #สามารถกำหนดค่า seq_length ใหม่ได้เพื่อทดลองว่า seq ไหนได้ผลดีที่สุด
vocab_size = 128 # ที่เป็นค่า 128 เพราะใช้แทน note ทั้งหมดที่รองรับใน pretty_midi
key_order = ['pitch', 'step', 'duration']

if submit:
    midi_data = pretty_midi.PrettyMIDI(getFile(partDataset + Original_song))
    for instrument in midi_data.instruments:
        instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
    raw_notes = midi_to_notes(getFile(partDataset + Original_song))

    temperature = 2.0

    sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)
    input_notes = (sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))

    model = tf.keras.models.load_model("./" + partModel + model_choice, custom_objects={'mse_with_positive_pressure': mse_with_positive_pressure})

    st.info('Please wait.')
    generated_notes = []
    prev_start = 0
    for _ in range(num_predictions):
        pitch, step, duration = predict_next_note(input_notes, model, temperature)
        start = prev_start + step
        end = start + duration
        input_note = (pitch, step, duration)
        generated_notes.append((*input_note, start, end))
        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
        prev_start = start

    generated_notes = pd.DataFrame(generated_notes, columns=(*key_order, 'start', 'end'))

    out_pm = notes_to_midi(generated_notes, "output.mid", instrument_name=instrument_name)
    st.markdown("---")
    text_summary = 'Generate From Music =>  ' + Original_song + ' ,  Artist =>  ' + model_choice + ' ,  Note =>  ' + str(num_predictions) + '.'
    st.success(text_summary)
    # play music
    fs = FluidSynth()
    fs.midi_to_audio('output.mid', 'output.wav')
    audio_file = open('output.wav', 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes)
    st.markdown("Download the audio by right-clicking on the media player")

st.markdown("---")
st.write("Upload midi file")
uploaded_files = st.file_uploader("Choose a midi file", type=["mid", "midi"])
if uploaded_files is not None:
  file_details = {"filename":uploaded_files.name,"filetype":uploaded_files.type,"filesize":uploaded_files.size}
  st.write(file_details)
  with open(os.path.join("Dataset",uploaded_files.name),"wb") as f:
    f.write((uploaded_files).getbuffer())
    st.success("File Midi Saved")
    file_names = [fn for fn in os.listdir(partDataset)]

st.markdown("---")
st.write("Upload model")
uploaded_file = st.file_uploader("Choose a model file", type=["h5"])
if uploaded_file is not None:
  file_details = {"filename":uploaded_file.name, "filetype":uploaded_file.type,"filesize":uploaded_file.size}
  st.write(file_details)
  with open(os.path.join("Model",uploaded_file.name),"wb") as f:
    f.write((uploaded_file).getbuffer())
    st.success("File Model Saved")
    model_names = [fn for fn in os.listdir(partModel)]