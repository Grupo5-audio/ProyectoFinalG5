import tensorflow as tf
from transformers import TFWav2Vec2Model, Wav2Vec2Processor
import librosa
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import os
import pandas as pd
import numpy as np
import librosa
import seaborn as sns
import matplotlib.pyplot as plt

import joblib


#cargamos los datos de la fuente de RAVDESS. Recibe como parámetro la ruta donde se encuentran los archivos.
def load_ravdess_dataset(ravdess_path):
    # Listar directorios
    ravdess_directory_list = os.listdir(ravdess_path)

    file_emotion = []
    file_path = []

    for dir in ravdess_directory_list:
        actor = os.listdir(os.path.join(ravdess_path, dir))
        for file in actor:
            part = file.split('.')[0].split('-')
            file_emotion.append(int(part[2]))
            file_path.append(os.path.join(ravdess_path, dir, file))

    # Crear DataFrame
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    df = pd.concat([emotion_df, path_df], axis=1)

    df.replace({'Emotions': {
        1: 'neutral', 2: 'neutral', 3: 'felicidad', 4: 'triste',
        5: 'enojado', 6: 'miedo', 7: 'desagrado', 8: 'sorprendido'
    }}, inplace=True)

    total = len(df)
    emociones_unicas = df['Emotions'].nunique()
    lista_emociones = df['Emotions'].unique().tolist()
    return df, total, emociones_unicas, lista_emociones

#cargamos los datos de la fuente de CREMA. Recibe como parámetro la ruta donde se encuentran los archivos.
def load_crema_dataset(crema_path):
    crema_directory_list = os.listdir(crema_path)
    file_emotion = []
    file_path = []

    for file in crema_directory_list:
        # storing file paths
        file_path.append(crema_path + file)
        # storing file emotions
        part=file.split('_')
        if part[2] == 'SAD':
            file_emotion.append('triste')
        elif part[2] == 'ANG':
            file_emotion.append('enojado')
        elif part[2] == 'DIS':
            file_emotion.append('desagrado')
        elif part[2] == 'FEA':
            file_emotion.append('miedo')
        elif part[2] == 'HAP':
            file_emotion.append('felicidad')
        elif part[2] == 'NEU':
            file_emotion.append('neutral')
        else:
            file_emotion.append('Unknown')

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    df = pd.concat([emotion_df, path_df], axis=1)

    total = len(df)
    emociones_unicas = df['Emotions'].nunique()
    lista_emociones = df['Emotions'].unique().tolist()
    return df, total, emociones_unicas, lista_emociones

#cargamos los datos de la fuente de TESS. Recibe como parámetro la ruta donde se encuentran los archivos.
def load_tess_dataset(tess_path):
    tess_directory_list = os.listdir(tess_path)

    file_emotion = []
    file_path = []

    for dir in tess_directory_list:
        directories = os.listdir(os.path.join(tess_path, dir))
        for file in directories:
            part = file.split('.')[0].split('_')[2]
            file_path.append(os.path.join(tess_path, dir, file))
            file_emotion.append(part)

    # Traducir emociones de inglés a español
    traduccion_emociones = {
        'fear': 'miedo',
        'angry': 'enojado',
        'disgust': 'desagrado',
        'neutral': 'neutral',
        'sad': 'triste',
        'ps': 'sorprendido',  # ps = pleasant surprise
        'happy': 'felicidad'
    }

    emociones_traducidas = [traduccion_emociones.get(e, e) for e in file_emotion]

    #crea data sintetica para mejorar la generalización del audio

    # Crear DataFrames
    emotion_df = pd.DataFrame(emociones_traducidas, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    df = pd.concat([emotion_df, path_df], axis=1)

    # Datos resumidos
    total = len(df)
    emociones_unicas = df['Emotions'].nunique()
    lista_emociones = df['Emotions'].unique().tolist()

    return df, total, emociones_unicas, lista_emociones

#cargamos los datos de la fuente de SAVEE. Recibe como parámetro la ruta donde se encuentran los archivos.
def load_savee_dataset(savee_path):
    savee_directory_list = os.listdir(savee_path)

    file_emotion = []
    file_path = []

    for file in savee_directory_list:
        file_path.append(savee_path + file)
        part = file.split('_')[1]
        ele = part[:-6]
        if ele=='a':
            file_emotion.append('enojado')
        elif ele=='d':
            file_emotion.append('desagrado')
        elif ele=='f':
            file_emotion.append('miedo')
        elif ele=='h':
            file_emotion.append('felicidad')
        elif ele=='n':
            file_emotion.append('neutral')
        elif ele=='sa':
            file_emotion.append('triste')
        else:
            file_emotion.append('sorprendido')

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    df = pd.concat([emotion_df, path_df], axis=1)

    total = len(df)
    emociones_unicas = df['Emotions'].nunique()
    lista_emociones = df['Emotions'].unique().tolist()
    return df, total, emociones_unicas, lista_emociones

#función que explora los datos. Crea un gráfico de barras con las emociones identificadas
def explore_data(df):
    df.to_csv("src/data_path.csv", index=False)
    #print("DF en explore_data")
    #print(df.head())

    plt.figure(figsize=(10, 6))
    plt.title('Conteo de Emociones', size=16)
    sns.countplot(data=df, x='Emotions', order=df['Emotions'].value_counts().index)
    plt.ylabel('Conteo', size=12)
    plt.xlabel('Emociones', size=12)
    sns.despine()
    plt.show()


def encode_labels(labels, save_path=None):
    encoder = OneHotEncoder()
    Y_encoded = encoder.fit_transform(np.array(labels).reshape(-1, 1)).toarray()
    # Guardar las clases si se proporciona una ruta
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, "class_labels.npy"), encoder.categories_[0])
        print(f"✅ Clases guardadas en: {os.path.join(save_path, 'class_labels.npy')}")

    return Y_encoded, encoder


# Cargar y procesar datos
def load_audio(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    return audio

def prepare_dataset(df, processor, sr=16000):
    input_values = []

    Y = []
    
    for path, emotion in zip(df.Path, df.Emotions):
        audio = load_audio(path, sr=sr)
        processed = processor(audio, sampling_rate=sr, return_tensors="tf", padding=True)
        input_values.append(processed.input_values[0])  # Solo la señal procesada
        Y.append(emotion)

    max_len = max([x.shape[0] for x in input_values])
    input_values = [tf.pad(x, paddings=[[0, max_len - tf.shape(x)[0]]]) for x in input_values]
    input_values = tf.stack(input_values)  # (n, max_len)

    y, encoder = encode_labels(Y)

    return input_values, y, encoder

""" # Tus datos
    audio_files = [
    "ruta/audio1.wav",
    "ruta/audio2.wav",
    # ...
]
labels = ["feliz", "triste", "enojado", ...]  # Deben coincidir
 """

def run_pipeline(ravdess_path=None, crema_path=None, tess_path=None, savee_path=None):

    # Cargar modelo y processor
    model_name = "facebook/wav2vec2-base"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    wav2vec_model = TFWav2Vec2Model.from_pretrained(model_name, from_pt=True)


    dfs = []

    if ravdess_path:
        df_ravdess, total_r, emociones_r, lista_emociones_r = load_ravdess_dataset(ravdess_path)
        print(f"[RAVDESS] Total de registros: {total_r}")
        print(f"[RAVDESS] Total de emociones únicas: {emociones_r}")
        print(f"[RAVDESS] Emociones presentes: {lista_emociones_r}")
        dfs.append(df_ravdess)
    
    if crema_path:
        df_crema, total_r, emociones_r, lista_emociones_r = load_crema_dataset(crema_path)
        print(f"[CREMA] Total de registros: {total_r}")
        print(f"[CREMA] Total de emociones únicas: {emociones_r}")
        print(f"[CREMA] Emociones presentes: {lista_emociones_r}")
        dfs.append(df_crema)

    if tess_path:
        df_tess, total_r, emociones_r, lista_emociones_r = load_tess_dataset(tess_path)
        print(f"[TESS] Total de registros: {total_r}")
        print(f"[TESS] Total de emociones únicas: {emociones_r}")
        print(f"[TESS] Emociones presentes: {lista_emociones_r}")
        dfs.append(df_tess)
    
    if savee_path:
        df_savee, total_r, emociones_r, lista_emociones_r = load_savee_dataset(savee_path)
        print(f"[SAVEE] Total de registros: {total_r}")
        print(f"[SAVEE] Total de emociones únicas: {emociones_r}")
        print(f"[SAVEE] Emociones presentes: {lista_emociones_r}")
        dfs.append(df_savee)

    # Combinar todos los datasets en uno solo
    full_df = pd.concat(dfs, ignore_index=True)
    
    full_df = full_df[~full_df['Emotions'].str.lower().str.contains("sorprendido")]

    print(full_df.head())

    print("Exploración de datos")
    explore_data(full_df)



    # Procesar dataset
    X, y, encoder = prepare_dataset(full_df, processor)

    if hasattr(X, "numpy"):
        X = X.numpy()
    else:
        X = np.array(X)

    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # Construcción del modelo funcional
    input_layer = tf.keras.Input(shape=(X_train.shape[1],), dtype=tf.float32, name="input_values")
    wav2vec_output = wav2vec_model(input_layer).last_hidden_state  # (batch, time, 768)
    x = tf.keras.layers.GlobalAveragePooling1D()(wav2vec_output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output_layer = tf.keras.layers.Dense(y.shape[1], activation='softmax')(x)

    final_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Congelar o descongelar parte de Wav2Vec2 (ajusta según memoria/disponibilidad)
    for layer in wav2vec_model.layers:
        layer.trainable = True  # ← descomenta para fine-tuning completo

    # Compilar modelo
    final_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    final_model.summary()

    # Entrenar
    final_model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=4)


    # 💾 Guardar modelo entrenado
    models_path="models/"
    model_path = os.path.join(models_path, "final_model.pkl")
    joblib.dump(final_model, model_path)
    print(f"📦 Modelo guardado en: {model_path}")
    #grafico_perdida(history)


Ravdess = "C:/Users/andra/.cache/kagglehub/datasets/uwrfkaggler/ravdess-emotional-speech-audio/versions/1/audio_speech_actors_01-24/"
Crema = "C:/Users/andra/.cache/kagglehub/datasets/ejlok1/cremad/versions/1/AudioWAV/"
Tess = "C:/Users/andra/.cache/kagglehub/datasets/ejlok1/toronto-emotional-speech-set-tess/versions/1/TESS Toronto emotional speech set data/TESS Toronto emotional speech set data/"
Savee = "C:/Users/andra/.cache/kagglehub/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee/versions/1/ALL/"

# Ejecutar pipeline
X, Y = run_pipeline(
    ravdess_path=Ravdess,
    crema_path=Crema,
    tess_path=Tess,
    savee_path=Savee
)


