import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent

data_dir = repo_root / 'Super Mario Bros. (World)' / 'Data'
output_file = repo_root / 'LuaScriptData' / 'InputGenerator' / 'mario_mlp.json'


def _sorted_files(pattern):
    files = list(data_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f'No files found for pattern {pattern} in {data_dir.resolve()}'
        )

    def _num(s):
        m = re.search(r'_(\d+)\.csv$', str(s))
        return int(m.group(1)) if m else 0

    files.sort(key=_num)
    return [str(p) for p in files]


def cargar_partidas(lista_archivos):
    X_total = []
    Y_total = []

    for archivo in lista_archivos:
        df = pd.read_csv(archivo)
        datos = df.values

        X = datos[:-1, :]
        Y = datos[1:, :6]

        X_total.append(X)
        Y_total.append(Y)

    X_total = np.vstack(X_total)
    Y_total = np.vstack(Y_total)
    return X_total, Y_total


def exportar_modelo_y_scaler(modelo, scaler, archivo_salida):
    datos = {
        "n_layers": modelo.n_layers_,
        "n_outputs": modelo.n_outputs_,
        "hidden_layer_sizes": (
            list(modelo.hidden_layer_sizes)
            if isinstance(modelo.hidden_layer_sizes, tuple)
            else [modelo.hidden_layer_sizes]
        ),
        "activation": modelo.activation,
        "out_activation": modelo.out_activation_,
        "coefs": [W.tolist() for W in modelo.coefs_],
        "biases": [b.tolist() for b in modelo.intercepts_],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }

    archivo_salida = Path(archivo_salida)
    archivo_salida.parent.mkdir(parents=True, exist_ok=True)

    with open(archivo_salida, 'w') as f:
        json.dump(datos, f)

    print(f'Modelo exportado a: {archivo_salida.resolve()}')


def main():
    train_files = _sorted_files('train_*.csv')
    test_files = _sorted_files('test_*.csv')

    print('train files:', train_files)
    print('test files :', test_files)

    X_train, Y_train = cargar_partidas(train_files)
    X_test, Y_test = cargar_partidas(test_files)

    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    modelo_final = MLPClassifier(
        hidden_layer_sizes=(125, 9, 6),
        activation='relu',
        solver='adam',
        batch_size=256,
        max_iter=1000,
        random_state=42,
    )

    modelo_final.fit(X_train, Y_train)

    test_score = modelo_final.score(X_test, Y_test)
    print(f'Accuracy test: {test_score:.4f}')

    exportar_modelo_y_scaler(
        modelo_final,
        scaler,
        output_file,
    )


if __name__ == '__main__':
    main()
