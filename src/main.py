import json
import re
from pathlib import Path
import time
import subprocess
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib
import os
from datetime import datetime
import shutil

horizon = 63

SCRIPT_DIR = Path(__file__).resolve().parent
repo_root = SCRIPT_DIR.parent

data_dir = repo_root / 'Super Mario Bros. (World)' / 'Data'
output_file = repo_root / 'LuaScriptData' / 'InputGenerator' / 'mario_mlp.json'
models_dir = repo_root / 'Super Mario Bros. (World)' / 'Models'

# Crear directorio de modelos si no existe
models_dir.mkdir(parents=True, exist_ok=True)

MESEN = repo_root / 'Super Mario Bros. (World)' / 'Mesen.exe'
ROM = repo_root / 'Super Mario Bros. (World)' / 'Super Mario Bros. (World).nes'
INPUT_SCRIPT = repo_root / 'Super Mario Bros. (World)' / 'DMP' / 'InputGenerator.lua'
TEST_SCRIPT = repo_root / 'Super Mario Bros. (World)' / 'DMP' / 'DataExtractor.lua'

class MarioMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 107),
            nn.ReLU(),
            nn.Linear(107, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class FitnessFutureMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 107),
            nn.ReLU(),
            nn.Linear(107, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_fitness_predictor_temporal(
    X,
    FITNESS,
    horizon=63,
    batch_size=256,
    lr=1e-3,
    epochs=100
):
    X = np.asarray(X)
    FITNESS = np.asarray(FITNESS)

    # Solo recortar si horizon > 0
    if horizon > 0:
        X_train = X[:-horizon]
        y_train = FITNESS[horizon:]
    else:
        X_train = X
        y_train = FITNESS

    print(f"[FitnessPredictor] Training data shapes: X={X_train.shape}, y={y_train.shape}")

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    dataset = TensorDataset(X_train, y_train)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = FitnessFutureMLP(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for x, y in loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[FitnessPredictor] Epoch {epoch} | Loss: {total_loss:.4f}")

    return model, X_scaler, y_scaler

def mostrar_resultados(results):
    print("\n" + "="*60)
    print("RESULTADOS DE LAS PRUEBAS")
    print("="*60)
    
    if "results" not in results or len(results["results"]) == 0:
        print("No hay resultados disponibles")
        return None
    
    for result in results["results"]:
        status_emoji = "✅" if result["status"] == "complete" else "❌"
        status_text = {
            "complete": "PASÓ",
            "failed": "FALLÓ",
            "timeout": "TIMEOUT",
            "incomplete": "INCOMPLETO"
        }.get(result["status"], result["status"])
        
        print(f"{status_emoji} Test {result['test_number']}: {status_text} - Fitness: {result['fitness']:.2f} - X: {result['final_x']:.0f}")
    
    total = len(results["results"])
    completados = sum(1 for r in results["results"] if r["status"] == "complete")
    fallados = sum(1 for r in results["results"] if r["status"] == "failed")
    timeouts = sum(1 for r in results["results"] if r["status"] == "timeout")
    
    print(f"\n📊 Estadísticas:")
    print(f"  Total: {total}")
    print(f"  ✅ Completados: {completados} ({completados/total*100:.1f}%)")
    print(f"  ❌ Fallados: {fallados} ({fallados/total*100:.1f}%)")
    print(f"  ⏱️ Timeouts: {timeouts} ({timeouts/total*100:.1f}%)")
    print("="*60)
    
    return {
        'total': total,
        'completados': completados,
        'fallados': fallados,
        'timeouts': timeouts,
        'pct_completados': completados / total * 100 if total > 0 else 0,
        'pct_fallados': fallados / total * 100 if total > 0 else 0
    }

def ejecutar_test_runner_con_monitoreo():
    TEST_RUNNER_SCRIPT = repo_root / 'Super Mario Bros. (World)' / 'DMP' / 'TestRunner.lua'
    RESULTS_FILE = data_dir / "test_results.json"
    
    if not TEST_RUNNER_SCRIPT.exists():
        print(f"ERROR: No se encontró TestRunner.lua en {TEST_RUNNER_SCRIPT}")
        return None
    
    print("\n" + "="*60)
    print("EJECUTANDO TEST RUNNER CON MONITOREO")
    print("="*60)
    
    # Borrar archivo de resultados anterior para detectar nueva escritura
    if RESULTS_FILE.exists():
        RESULTS_FILE.unlink()
        print("Archivo de resultados anterior eliminado")
    
    try:
        proceso = subprocess.Popen([
            str(MESEN),
            str(ROM),
            str(TEST_RUNNER_SCRIPT)
        ], shell=False)
        
        print(f"Mesen iniciado con TestRunner (PID: {proceso.pid})")
        print("Monitoreando archivo de resultados...")
        
        max_wait_time = 150  # 2 minutos máximo
        start_time = time.time()
        check_interval = 1
        last_mtime = None
        
        while True:
            # Verificar si el proceso sigue vivo
            if proceso.poll() is not None:
                print(f"❌ Mesen se cerró inesperadamente (código: {proceso.returncode})")
                break
            
            # Verificar timeout total
            elapsed = time.time() - start_time
            if elapsed > max_wait_time:
                print(f"⚠️ Timeout: {max_wait_time}s excedidos")
                print("🔴 Cerrando Mesen...")
                proceso.terminate()
                try:
                    proceso.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proceso.kill()
                    proceso.wait()
                break
            
            # Verificar si el archivo de resultados fue modificado
            if RESULTS_FILE.exists():
                current_mtime = RESULTS_FILE.stat().st_mtime
                
                if current_mtime != last_mtime:
                    last_mtime = current_mtime
                    
                    # Pequeña espera para asegurar que se termine de escribir
                    time.sleep(1)
                    
                    try:
                        with open(RESULTS_FILE, "r") as f:
                            content = f.read()
                            if not content.strip():
                                continue
                            
                            results = json.loads(content)
                            
                        if "results" in results and len(results["results"]) > 0:
                            print(f"✅ Resultados válidos: {len(results['results'])} tests")
                            
                            # Cerrar Mesen
                            print("🔴 Cerrando Mesen...")
                            proceso.terminate()
                            
                            try:
                                proceso.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                proceso.kill()
                                proceso.wait()
                            
                            print("Mesen cerrado correctamente")
                            return results
                            
                    except (json.JSONDecodeError, Exception):
                        # Archivo en escritura o corrupto, seguir esperando
                        pass
            
            time.sleep(check_interval)
        
        # Si salimos del bucle, intentar recuperar resultados parciales
        if RESULTS_FILE.exists():
            try:
                with open(RESULTS_FILE, "r") as f:
                    results = json.load(f)
                    if "results" in results and len(results["results"]) > 0:
                        print(f"📊 Recuperados {len(results['results'])} resultados parciales")
                        return results
            except:
                pass
        
        print("No se detectaron resultados válidos")
        return None
            
    except Exception as e:
        print(f"Error al ejecutar TestRunner: {e}")
        if 'proceso' in locals():
            try:
                proceso.terminate()
                proceso.wait(timeout=5)
            except:
                try:
                    proceso.kill()
                except:
                    pass
        return None

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

def visualizar_progreso(history):
    """Muestra una tabla con el progreso de los porcentajes de completado"""
    if not history:
        return
    
    print("\n" + "="*80)
    print("📈 EVOLUCIÓN DE PORCENTAJES DE COMPLETADO")
    print("="*80)
    print(f"{'Iter':^6} | {'Alpha':^8} | {'Promedio':^8} | {'Completados':^10} | {'Niveles (>50%)'} | {'Progreso'}")
    print("-"*80)
    
    for h in history:
        iter_num = h['iteration']
        alpha = h['alpha']
        promedio = h.get('porcentaje_promedio', 0)
        completados = h['stats']['pct_completados']
        
        # Contar niveles con >50% completado
        niveles_buenos = sum(1 for p in h.get('porcentajes_nivel', []) if p['porcentaje'] > 50)
        total_niveles = len(h.get('porcentajes_nivel', []))
        
        # Barra de progreso simple
        barra = '█' * int(promedio / 5) + '░' * (20 - int(promedio / 5))
        
        print(f"{iter_num:^6} | {alpha:^8.4f} | {promedio:^8.1f}% | {completados:^10.1f}% | {niveles_buenos}/{total_niveles} | {barra}")
    
    print("="*80)

def cargar_partidas(lista_archivos):
    X_total = []
    Y_total = []
    F_total = []
    for archivo in lista_archivos:
        df = pd.read_csv(archivo)
        fitness = df["Fitness"].values
        df_features = df.drop(columns=["Fitness"])
        datos = df_features.values
        X = datos[:-1, :]
        Y = datos[1:, :6]
        F = fitness[:-1]
        X_total.append(X)
        Y_total.append(Y)
        F_total.append(F)
    X_total = np.vstack(X_total)
    Y_total = np.vstack(Y_total)
    F_total = np.concatenate(F_total)
    return X_total, Y_total, F_total

def calcular_porcentaje_completado(resultados_test, marioX_inicial=None):
    """
    Calcula el porcentaje de completado para cada nivel basado en la posición de Mario.
    El porcentaje = (posicion_final - posicion_inicial) / (4128 - posicion_inicial) * 100
    Donde 4128 es la posición en píxeles de la bandera (X = 258 bloques * 16 píxeles)
    """
    porcentajes = []
    
    for result in resultados_test:
        if 'final_x' in result and result['final_x'] > 0:
            # Si no tenemos la posición inicial, intentamos obtenerla del resultado
            x_inicial = result.get('initial_x', marioX_inicial)
            if x_inicial is None:
                x_inicial = 50  # Valor por defecto
            x_final = result['final_x']
            
            # Posición de la bandera en píxeles (258 bloques * 16 píxeles)
            FLAG_POSITION = 4128
            
            # Calcular porcentaje
            if x_final > x_inicial:
                distancia_recorrida = x_final - x_inicial
                distancia_total = FLAG_POSITION - x_inicial
                porcentaje = (distancia_recorrida / distancia_total) * 100
                # Cap en 100%
                porcentaje = min(porcentaje, 100)
            else:
                porcentaje = 0
            
            # Si el nivel está completado según el status, forzar 100%
            if result.get('status') == 'complete':
                porcentaje = 100
            
            porcentajes.append({
                'test_number': result.get('test_number', 0),
                'porcentaje': porcentaje,
                'x_final': x_final,
                'x_inicial': x_inicial,
                'status': result.get('status', 'unknown')
            })
    
    return porcentajes

def guardar_historial_csv(history, models_dir):
    """Guarda el historial en formato CSV para análisis en Excel/Google Sheets"""
    import csv
    
    csv_path = models_dir / "historial_entrenamiento.csv"
    
    # Determinar cuántos niveles hay
    max_niveles = 0
    for h in history:
        if 'porcentajes_nivel' in h:
            max_niveles = max(max_niveles, len(h['porcentajes_nivel']))
    
    # Crear encabezados
    header = ['Iteracion', 'Alpha', 'Porcentaje_Promedio', 'Completados_Pct', 
              'Fallados_Pct', 'Fitness_Avg']
    
    for i in range(max_niveles):
        header.append(f'Nivel_{i+1}')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for h in history:
            row = [
                h['iteration'],
                h['alpha'],
                h['porcentaje_promedio'],
                h['stats']['pct_completados'],
                h['stats']['pct_fallados'],
                h['stats'].get('avg_fitness', 0)
            ]
            
            # Agregar porcentaje por nivel
            for p in h.get('porcentajes_nivel', []):
                row.append(p['porcentaje'])
            
            # Completar con ceros si hay menos niveles
            while len(row) < len(header):
                row.append(0)
            
            writer.writerow(row)
    
    print(f"📊 Historial CSV guardado en: {csv_path}")

def guardar_historial_completo(history, models_dir):
    """Guarda el historial completo incluyendo porcentajes por nivel"""
    historial_path = models_dir / "historial_entrenamiento.json"
    
    # Preparar datos para JSON
    historial_data = []
    for h in history:
        # Convertir a formato serializable
        entry = {
            'iteration': h['iteration'],
            'alpha': h['alpha'],
            'porcentaje_promedio': h['porcentaje_promedio'],
            'completados': h['stats']['pct_completados'],
            'fallados': h['stats']['pct_fallados'],
            'avg_fitness': h['stats'].get('avg_fitness', 0),
            'porcentajes_nivel': h['porcentajes_nivel']
        }
        historial_data.append(entry)
    
    with open(historial_path, 'w') as f:
        json.dump(historial_data, f, indent=2)
    
    print(f"📊 Historial completo guardado en: {historial_path}")
    return historial_path

def generar_graficas_evolucion(history, models_dir):
    """Genera gráficas de evolución de los porcentajes de completado"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        iterations = [h['iteration'] for h in history]
        porcentajes_promedio = [h.get('porcentaje_promedio', 0) for h in history]
        completados = [h['stats']['pct_completados'] for h in history]
        avg_fitness = [h['stats'].get('avg_fitness', 0) for h in history]
        alphas = [h['alpha'] for h in history]
        
        # Crear figura con múltiples subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Gráfica 1: Porcentaje de completado promedio
        axes[0, 0].plot(iterations, porcentajes_promedio, 'b-o', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Iteración', fontsize=12)
        axes[0, 0].set_ylabel('Porcentaje completado promedio (%)', fontsize=12)
        axes[0, 0].set_title('Evolución del porcentaje de completado', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 105)
        
        # Gráfica 2: Tests completados vs fallados
        axes[0, 1].plot(iterations, completados, 'g-o', linewidth=2, markersize=8, label='Completados')
        axes[0, 1].plot(iterations, [h['stats']['pct_fallados'] for h in history], 'r-o', linewidth=2, markersize=8, label='Fallados')
        axes[0, 1].set_xlabel('Iteración', fontsize=12)
        axes[0, 1].set_ylabel('Porcentaje (%)', fontsize=12)
        axes[0, 1].set_title('Evolución de tests completados vs fallados', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 105)
        
        # Gráfica 3: Fitness promedio
        axes[1, 0].plot(iterations, avg_fitness, 'm-o', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Iteración', fontsize=12)
        axes[1, 0].set_ylabel('Fitness promedio', fontsize=12)
        axes[1, 0].set_title('Evolución del fitness promedio', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Gráfica 4: Alpha
        axes[1, 1].plot(iterations, alphas, 'orange-o', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('Iteración', fontsize=12)
        axes[1, 1].set_ylabel('Alpha', fontsize=12)
        axes[1, 1].set_title('Evolución de Alpha', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar gráfica
        graph_path = models_dir / "evolucion_entrenamiento.png"
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        print(f"📊 Gráfica guardada en: {graph_path}")
        
        # Gráfica simplificada del porcentaje de completado
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(iterations, porcentajes_promedio, 'b-o', linewidth=2, markersize=8)
        
        if len(porcentajes_promedio) > 0:
            max_idx = np.argmax(porcentajes_promedio)
            ax2.plot(iterations[max_idx], porcentajes_promedio[max_idx], 'r*', markersize=15, 
                    label=f'Máximo: {porcentajes_promedio[max_idx]:.1f}% (Iter {iterations[max_idx]})')
        
        ax2.set_xlabel('Iteración', fontsize=12)
        ax2.set_ylabel('Porcentaje de completado promedio (%)', fontsize=12)
        ax2.set_title('Evolución del porcentaje de completado de niveles', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0, 105)
        
        simple_path = models_dir / "porcentaje_completado_evolucion.png"
        plt.savefig(simple_path, dpi=150, bbox_inches='tight')
        print(f"📊 Gráfica simple guardada en: {simple_path}")
        
        plt.close('all')
        return graph_path
        
    except ImportError:
        print("⚠️ matplotlib no instalado. No se pueden generar gráficas.")
        return None
    except Exception as e:
        print(f"⚠️ Error al generar gráficas: {e}")
        return None

def exportar_modelo_y_scaler(modelo, scaler, archivo_salida):
    archivo_salida = Path(archivo_salida)
    archivo_salida.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        k: v.detach().cpu().tolist()
        for k, v in modelo.state_dict().items()
    }
    datos = {
        "model_type": modelo.__class__.__name__,
        "state_dict": state_dict,
        "input_dim": list(modelo.parameters())[0].shape[1],
        "output_dim": list(modelo.parameters())[-1].shape[0],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    with open(archivo_salida, "w") as f:
        json.dump(datos, f)
    print(f"Modelo exportado a: {archivo_salida.resolve()}")

def guardar_modelo(model, scaler, iter_num, alpha, stats):
    """Guarda el modelo con información de la iteración"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = models_dir / f"model_iter_{iter_num:04d}_{timestamp}.pt"
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_dim': model.net[0].in_features,
        'output_dim': model.net[-1].out_features,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'iter_num': iter_num,
        'alpha': alpha,
        'timestamp': timestamp,
        'stats': stats
    }
    
    torch.save(checkpoint, filename)
    print(f"Modelo guardado en: {filename}")
    
    # Guardar como último modelo
    latest_path = models_dir / "latest_model.pt"
    torch.save(checkpoint, latest_path)
    
    return filename

def cargar_ultimo_modelo():
    """Carga el último modelo guardado si existe"""
    latest_path = models_dir / "latest_model.pt"
    if not latest_path.exists():
        return None, None, 0, 0.01, None
    
    try:
        checkpoint = torch.load(latest_path, weights_only=False)
        model = MarioMLP(checkpoint['input_dim'], checkpoint['output_dim'])
        model.load_state_dict(checkpoint['model_state_dict'])
        
        scaler = StandardScaler()
        scaler.mean_ = np.array(checkpoint['scaler_mean'])
        scaler.scale_ = np.array(checkpoint['scaler_scale'])
        
        iter_num = checkpoint.get('iter_num', 0)
        alpha = checkpoint.get('alpha', 0.01)
        stats = checkpoint.get('stats', None)
        
        print(f"Modelo cargado desde iteración {iter_num} con alpha={alpha}")
        return model, scaler, iter_num, alpha, stats
    except Exception as e:
        print(f"Error al cargar modelo: {e}")
        return None, None, 0, 0.01, None

def exportar_mejor_modelo():
    best_path = models_dir / "model_best.pt"

    if not best_path.exists():
        print("⚠️ No existe model_best.pt")
        return

    checkpoint = torch.load(best_path, weights_only=False)

    model = MarioMLP(
        checkpoint['input_dim'],
        checkpoint['output_dim']
    )

    model.load_state_dict(
        checkpoint['model_state_dict']
    )

    scaler = StandardScaler()
    scaler.mean_ = np.array(checkpoint['scaler_mean'])
    scaler.scale_ = np.array(checkpoint['scaler_scale'])

    exportar_modelo_y_scaler(
        model,
        scaler,
        output_file
    )

    print("✅ Mejor modelo exportado a mario_mlp.json")

def entrenar_iterativo():
    """
    Entrenamiento iterativo con ajuste de alpha basado en resultados de tests.
    """
    print("\n" + "="*70)
    print("🚀 INICIANDO ENTRENAMIENTO ITERATIVO CON AJUSTE DE ALPHA")
    print("="*70)
    
    # Cargar datos de entrenamiento
    train_files = _sorted_files('train_*.csv')
    print(f'Archivos de entrenamiento: {train_files}')
    
    X_train, Y_train, F_train = cargar_partidas(train_files)
    print(f"Datos cargados: X={X_train.shape}, Y={Y_train.shape}, F={F_train.shape}")
    
    # Escalar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # ==============================================
    # ENTRENAR PREDICTOR DE FITNESS UNA SOLA VEZ
    # ==============================================
    print("\n" + "="*70)
    print("🔮 ENTRENANDO PREDICTOR DE FITNESS (UNA SOLA VEZ)")
    print("="*70)
    
    X_for_fitness = X_train_scaled[:-horizon]
    F_for_fitness = F_train[horizon:]
    
    print(f"Datos para predictor de fitness:")
    print(f"  X_for_fitness: {X_for_fitness.shape}")
    print(f"  F_for_fitness: {F_for_fitness.shape}")
    
    fitness_model, fitness_X_scaler, fitness_y_scaler = train_fitness_predictor_temporal(
        X=X_for_fitness,
        FITNESS=F_for_fitness,
        horizon=0,
        epochs=200
    )
    
    fitness_checkpoint = {
        'model_state_dict': fitness_model.state_dict(),
        'input_dim': list(fitness_model.parameters())[0].shape[1],
        'X_scaler_mean': fitness_X_scaler.mean_.tolist(),
        'X_scaler_scale': fitness_X_scaler.scale_.tolist(),
        'y_scaler_mean': fitness_y_scaler.mean_.tolist(),
        'y_scaler_scale': fitness_y_scaler.scale_.tolist(),
    }
    fitness_path = models_dir / "fitness_predictor.pt"
    torch.save(fitness_checkpoint, fitness_path)
    print(f"Predictor de fitness guardado en: {fitness_path}")
    
    # ==============================================
    # PREPARAR ESTRUCTURA DE DATOS PARA ENTRENAMIENTO
    # ==============================================
    
    max_t = len(X_train_scaled) - horizon - 1
    
    X_current_aligned = X_train_scaled[:max_t]
    X_next_aligned = X_train_scaled[1:max_t+1]
    Y_aligned = Y_train[:max_t]
    F_future_aligned = F_train[horizon:horizon+max_t]
    
    print(f"\nDatos alineados para entrenamiento:")
    print(f"  X_current (t): {X_current_aligned.shape}")
    print(f"  X_next (t+1): {X_next_aligned.shape}")
    print(f"  Y (inputs t+1): {Y_aligned.shape}")
    print(f"  F_future (t+horizon): {F_future_aligned.shape}")
    
    X_current_tensor = torch.tensor(X_current_aligned, dtype=torch.float32)
    Y_aligned_tensor = torch.tensor(Y_aligned, dtype=torch.float32)
    
    # Intentar cargar modelo previo
    model, loaded_scaler, start_iter, alpha, prev_stats = cargar_ultimo_modelo()
    
    if model is None:
        model = MarioMLP(
            input_dim=X_current_tensor.shape[1],
            output_dim=6
        )
        start_iter = 0
        alpha = 0.01
        print("Modelo nuevo creado")
    else:
        scaler = loaded_scaler
        print(f"Modelo cargado desde iteración {start_iter} con alpha={alpha}")
       
    # Parámetros de ajuste
    max_iterations = 3
    learning_rate = 1e-3
    best_completados = 0
    best_alpha = alpha
    best_fitness_seen = 1.0
    best_score = 0.0
    no_improve_count = 0
    patience = 3
    batch_size = 256
    
    ALPHA_MIN = 0.001
    ALPHA_MAX = 5.0
    
    history = []
    
    iteration = start_iter
    iterations_to_run = 3
    target_iteration = start_iter + iterations_to_run
    
    while iteration < target_iteration:
        iteration += 1
        print(f"\n{'='*70}")
        print(f"🔄 ITERACIÓN {iteration}/{max_iterations}")
        print(f"   Alpha actual: {alpha:.4f}")
        print(f"{'='*70}")
        
        # Entrenar el modelo
        print("\n📚 Entrenando modelo...")
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        
        epochs_per_iter = 200
        for epoch in range(epochs_per_iter):
            total_loss = 0
            total_loss_imitation = 0
            total_loss_fitness = 0
            
            dataset = TensorDataset(X_current_tensor, Y_aligned_tensor)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False
            )
            
            for batch_idx, (x_current, y_humano) in enumerate(loader):
                optimizer.zero_grad()
                
                y_pred = model(x_current)
                loss_imitation = loss_fn(y_pred, y_humano)
                
                start_idx = batch_idx * batch_size
                end_idx = start_idx + len(x_current)
                
                X_next_batch = X_next_aligned[start_idx:end_idx]
                F_future_batch = F_future_aligned[start_idx:end_idx]
                
                X_next_modified = X_next_batch.copy()
                X_next_modified[:, :6] = y_pred.detach().cpu().numpy()
                
                X_next_modified_scaled = fitness_X_scaler.transform(X_next_modified)
                X_next_modified_tensor = torch.tensor(
                    X_next_modified_scaled, 
                    dtype=torch.float32
                )
                
                with torch.no_grad():
                    fitness_pred_scaled = fitness_model(X_next_modified_tensor)
                
                fitness_pred = fitness_y_scaler.inverse_transform(
                    fitness_pred_scaled.cpu().numpy().reshape(-1, 1)
                ).flatten()
                
                fitness_pred_tensor = torch.tensor(fitness_pred, dtype=torch.float32)
                fitness_real_tensor = torch.tensor(F_future_batch, dtype=torch.float32)
                
                fitness_diff = fitness_real_tensor - fitness_pred_tensor
                loss_fitness = torch.mean(
                    torch.nn.functional.softplus(fitness_diff)
                )
                
                loss = loss_imitation + alpha * loss_fitness
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_loss_imitation += loss_imitation.item()
                total_loss_fitness += loss_fitness.item()
            
            if epoch % 10 == 0 or epoch == epochs_per_iter - 1:
                print(f"  Epoch {epoch+1}/{epochs_per_iter} | "
                      f"Total: {total_loss:.4f} | "
                      f"Imitación: {total_loss_imitation:.4f} | "
                      f"Fitness: {total_loss_fitness:.4f} | "
                      f"Alpha: {alpha:.4f}")
        
        # Exportar modelo para pruebas
        exportar_modelo_y_scaler(model, scaler, output_file)
        
        # Ejecutar pruebas
        print("\n🧪 Ejecutando pruebas...")
        results = ejecutar_test_runner_con_monitoreo()
        
        # ==============================================
        # MANEJAR CASO DE RESULTS = NONE
        # ==============================================
        if results is None:
            print("❌ Error en las pruebas. No se obtuvieron resultados.")
            print("   Posibles causas:")
            print("   - El TestRunner.lua tiene errores")
            print("   - Los archivos test_*.mss no existen o están corruptos")
            print("   - Mesen no pudo cargar correctamente")
            print("   - El modelo exportado tiene formato incorrecto")
            
            # Continuar con la siguiente iteración
            # Ajustar alpha para intentar mejorar
            alpha *= 1.1
            alpha = np.clip(alpha, ALPHA_MIN, ALPHA_MAX)
            print(f"🔄 Ajustando alpha a {alpha:.4f} para la siguiente iteración")
            continue
        
        # ==============================================
        # PROCESAR RESULTADOS VÁLIDOS
        # ==============================================
        
        # Calcular porcentajes de completado
        porcentajes_nivel = calcular_porcentaje_completado(results.get('results', []))
        porcentaje_promedio = np.mean([p['porcentaje'] for p in porcentajes_nivel]) if porcentajes_nivel else 0
        
        # Analizar resultados
        stats = mostrar_resultados(results)
        if stats is None:
            print("❌ No se pudieron analizar los resultados")
            continue

        avg_fitness = np.mean(
            [r['fitness'] for r in results['results']]
        )
        
        best_fitness_seen = max(best_fitness_seen, avg_fitness)
        fitness_score = np.log1p(avg_fitness) / np.log1p(best_fitness_seen)
        stats["avg_fitness"] = avg_fitness
        stats["fitness_score"] = fitness_score

        pct_completados = stats['pct_completados']
        pct_fallados = stats['pct_fallados']
        
        # Guardar en historial
        history.append({
            'iteration': iteration,
            'alpha': alpha,
            'stats': stats,
            'porcentajes_nivel': porcentajes_nivel,
            'porcentaje_promedio': porcentaje_promedio,
            'timestamp': datetime.now().isoformat()
        })
        
        # Guardar modelo
        guardar_modelo(model, scaler, iteration, alpha, stats)
        
        # ==============================================
        # AJUSTE DE ALPHA BASADO EN RESULTADOS
        # ==============================================
        
        old_alpha = alpha
        
        if pct_completados == 100:
            print("\n" + "="*70)
            print("🎉 ¡TODOS LOS TESTS PASARON! 🎉")
            print("="*70)
            final_path = models_dir / "model_final.pt"
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'input_dim': model.net[0].in_features,
                'output_dim': model.net[-1].out_features,
                'scaler_mean': scaler.mean_.tolist(),
                'scaler_scale': scaler.scale_.tolist(),
                'iter_num': iteration,
                'alpha': alpha,
                'stats': stats,
                'history': history
            }
            torch.save(checkpoint, final_path)
            print(f"🎯 Modelo final guardado en: {final_path}")
            return model, results, history
        
        # Estrategia basada en completados vs fallados
        if pct_completados >= 90:
            alpha *= 0.95
            print(f"🎯 Rendimiento excelente. Estabilizando con alpha={alpha:.4f}")
        elif pct_completados >= 70:
            alpha *= 1.05
            print(f"📈 Buen rendimiento. Aumentando alpha para mejorar: {alpha:.4f}")
        elif pct_completados >= 40:
            alpha *= 1.15
            print(f"⚠️ Rendimiento medio. Aumento agresivo de alpha: {alpha:.4f}")
        else:
            alpha *= 1.25
            learning_rate *= 0.8
            print(f"🚨 Bajo rendimiento. Alpha={alpha:.4f}, LR={learning_rate}")
            
        alpha = np.clip(alpha, ALPHA_MIN, ALPHA_MAX)
        
        if fitness_score > best_score:
            best_score = fitness_score
            best_completados = pct_completados
            best_alpha = old_alpha
            no_improve_count = 0
            print(f"✨ Nuevo mejor resultado: {pct_completados:.1f}% completados")
            
            best_path = models_dir / "model_best.pt"
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'input_dim': model.net[0].in_features,
                'output_dim': model.net[-1].out_features,
                'scaler_mean': scaler.mean_.tolist(),
                'scaler_scale': scaler.scale_.tolist(),
                'iter_num': iteration,
                'alpha': old_alpha,
                'stats': stats
            }
            torch.save(checkpoint, best_path)
            print(f"🏆 Mejor modelo guardado en: {best_path}")
        else:
            no_improve_count += 1
            print(f"⚠️ Sin mejora ({no_improve_count}/{patience} intentos)")
        
        if no_improve_count >= patience:
            learning_rate *= 0.5
            print(f"📉 Reduciendo learning rate a {learning_rate}")
            no_improve_count = 0
        
        print(f"\n📊 Resumen iteración {iteration}:")
        print(f"   Completados: {pct_completados:.1f}%")
        print(f"   Fallados: {pct_fallados:.1f}%")
        print(f"   Fitness media: {avg_fitness:.2f}")
        print(f"   Fitness score: {fitness_score:.4f}")
        print(f"   Alpha: {old_alpha:.4f} -> {alpha:.4f}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Porcentaje completado promedio: {porcentaje_promedio:.1f}%")
    
    # Al final del entrenamiento
    if history:
        print("\n" + "="*70)
        print("📊 GUARDANDO HISTORIAL Y GENERANDO GRÁFICAS")
        print("="*70)
        
        # Guardar historial en JSON y CSV
        guardar_historial_completo(history, models_dir)
        guardar_historial_csv(history, models_dir)
        
        # Mostrar visualización de progreso
        visualizar_progreso(history)
        
        # Generar gráficas
        generar_graficas_evolucion(history, models_dir)
    
    exportar_mejor_modelo()
    print(f"\n⚠️ Se alcanzó el máximo de iteraciones ({max_iterations})")
    return model, None, history

def main():
    model, results, history = entrenar_iterativo()

    if results is not None and history:
        print("\n✅ Entrenamiento completado con éxito!")
        print(f"📊 Historial de {len(history)} iteraciones:")

        for h in history:
            porcentaje = h.get('porcentaje_promedio', 0)
            print(
                f"  Iter {h['iteration']}: "
                f"alpha={h['alpha']:.4f}, "
                f"completados={h['stats']['pct_completados']:.1f}%, "
                f"porcentaje_niveles={porcentaje:.1f}%"
            )
        
        # Mostrar resultados finales
        print("\n📊 MEJORES RESULTADOS:")
        mejor_iter = max(history, key=lambda x: x.get('porcentaje_promedio', 0))
        print(f"  Mejor porcentaje de completado: {mejor_iter['porcentaje_promedio']:.1f}% (Iter {mejor_iter['iteration']})")
        
        ultima_iter = history[-1]
        print(f"\n  Última iteración ({ultima_iter['iteration']}):")
        for p in ultima_iter.get('porcentajes_nivel', []):
            status_emoji = "✅" if p['status'] == 'complete' else "❌"
            print(f"    {status_emoji} Nivel {p['test_number']}: {p['porcentaje']:.1f}% (X: {p['x_final']:.0f})")
            
    else:
        print("\n⚠️ Entrenamiento completado sin resultados.")
        if history:
            print(f"Se completaron {len(history)} iteraciones pero no se obtuvieron resultados válidos.")
            print("Posibles causas:")
            print("  - Problemas con la ejecución de Mesen")
            print("  - Archivos de prueba (test_*.mss) corruptos")
            print("  - Configuración incorrecta de rutas")

if __name__ == "__main__":
    main()
    #exportar_mejor_modelo()