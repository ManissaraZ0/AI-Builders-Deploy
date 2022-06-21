import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import collections
import pretty_midi
import tensorflow as tf
import os
import io
from scipy.io import wavfile

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

_SAMPLING_RATE = 16000
partModel = 'Model/'

st.title('👉 Classical Music Generator 👈')

Original_song = st.selectbox('Please select classical music', [
    'Albéniz','Bach','Beethoven','Brahms','Chopin','Clementi','Debussy','Grieg','Haydn','Mendelssohn','Mozart','Mussorgsky','Schubert','Tchaikovsky'], key='1')

partDataset = 'Dataset/'
if Original_song == 'Albéniz':
    inputs_ = 'Albéniz.mid'
if Original_song == 'Bach':
    inputs_ = 'Bach.mid'
if Original_song == 'Beethoven':
    inputs_ = 'Beethoven.mid'
if Original_song == 'Brahms':
    inputs_ = 'Brahms.mid'
if Original_song == 'Chopin':
    inputs_ = 'Chopin.mid'
if Original_song == 'Clementi':
    inputs_ = 'Clementi.mid'
if Original_song == 'Debussy':
    inputs_ = 'Debussy.mid'
if Original_song == 'Grieg':
    inputs_ = 'Grieg.mid'
if Original_song == 'Haydn':
    inputs_ = 'Haydn.mid'
if Original_song == 'Mendelssohn':
    inputs_ = 'Mendelssohn.mid'
if Original_song == 'Mozart':
    inputs_ = 'Mozart.mid'
if Original_song == 'Mussorgsky':
    inputs_ = 'Mussorgsky.mid'
if Original_song == 'Schubert':
    inputs_ = 'Schubert.mid'
if Original_song == 'Tchaikovsky':
    inputs_ = 'Tchaikovsky.mid'

model_choice = st.selectbox('Select your desired artist', [
    'Albéniz','Bach','Beethoven','Brahms','Chopin','Clementi','Debussy','Grieg','Haydn','Mendelssohn','Mozart','Mussorgsky','Schubert','Tchaikovsky'], key='1')

num_predictions = st.slider( "Sequence Lenght of Note", min_value=100 , max_value=500 ,value=300, step=1)

submit = st.button('Generate Classical Music')

seq_length = 50 #สามารถกำหนดค่า seq_length ใหม่ได้เพื่อทดลองว่า seq ไหนได้ผลดีที่สุด
vocab_size = 128 # ที่เป็นค่า 128 เพราะใช้แทน note ทั้งหมดที่รองรับใน pretty_midi
key_order = ['pitch', 'step', 'duration']

if submit:
    midi_data = pretty_midi.PrettyMIDI(getFile(partDataset + inputs_))
    for instrument in midi_data.instruments:
        instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
    raw_notes = midi_to_notes(getFile(partDataset + inputs_))

    temperature = 2.0

    sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)
    input_notes = (sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))

    st.success('Already Load Midi song!')
    model = tf.keras.models.load_model("./" + partModel + model_choice + '.h5', custom_objects={'mse_with_positive_pressure': mse_with_positive_pressure})
    st.success('Already Load Model!')

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
    st.success('Congratulation!')
    st.markdown("---")
    # play music
    st.info(out_pm.fluidsynth())
    with st.spinner(f"Transcribing to FluidSynth"):
        audio_data = out_pm.fluidsynth()
        audio_data = np.int16(
            audio_data / np.max(np.abs(audio_data)) * 32767 * 0.9
        )  # -- Normalize for 16 bit audio https://github.com/jkanner/streamlit-audio/blob/main/helper.py

        virtualfile = io.BytesIO()
        wavfile.write(virtualfile, 44100, audio_data)
    st.audio(virtualfile)
    st.markdown("Download the audio by right-clicking on the media player")