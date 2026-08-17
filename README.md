# Transformer desde Cero

Implementación de un **Transformer Encoder** para clasificación de sentimiento y un
**Mini-GPT** autoregresivo, escritos desde cero en PyTorch sin usar
`nn.MultiheadAttention`, `nn.TransformerEncoder` ni ninguna capa de alto nivel. Incluye
un artifact interactivo en React que reimplementa el forward pass completo en JavaScript
puro y corre con los pesos entrenados.

Proyecto 1 — Deep Learning, Universidad del Valle de Guatemala.

---

## Demo en video

[Ver la demo](https://youtu.be/D04I5UN1DHI)

Recorrido de 2 minutos por el artifact: análisis de sentimiento con mapas de atención,
generación de texto a dos temperaturas y discusión de resultados.

---

## Qué hay aquí

| Archivo | Contenido |
|---|---|
| `S6_Proyecto1.ipynb` | Notebook principal: ambos modelos, entrenamiento y análisis |
| `TransformerViz.jsx` | Artifact en React con el forward pass reimplementado en JS |
| `encoder_weights.json` | Pesos entrenados del encoder (15 tensores) |
| `gpt_weights.json` | Pesos entrenados del Mini-GPT (15 tensores) |

## Arquitectura

Ambos modelos comparten el mismo bloque base, implementado a mano:

- Embeddings aprendidos + codificación posicional sinusoidal
- Atención multi-cabeza con proyecciones $W_Q$, $W_K$, $W_V$, $W_O$ separadas
- Escalado por $\sqrt{d_k}$ y softmax sobre la última dimensión
- Conexiones residuales y Layer Normalization (media y varianza calculadas manualmente,
  con `unbiased=False`)
- Feed-forward de dos capas con ReLU

| Hiperparámetro | Encoder | Mini-GPT |
|---|---|---|
| Vocabulario | 328 | 432 |
| `d_model` | 32 | 32 |
| `d_ff` | 64 | 64 |
| Cabezas | 2 | 2 |
| Longitud máxima | 16 | 14 |
| Máscara | padding | causal + padding |
| Agregación | token `[CLS]` | logits por posición |

**Total de parámetros del Mini-GPT: 35,840** (sin contar biases). El Transformer base de
Vaswani et al. tiene ~62.98M, unas **1,760 veces más**. El 77% de los parámetros del
Mini-GPT vive en las tablas de embeddings; el bloque de atención completo son 4,096.

## Datos

SST-2 filtrado a oraciones de 3–15 palabras. 300 ejemplos de entrenamiento y 100 de
validación para clasificación; 400 oraciones para el modelo de lenguaje. Vocabulario
construido con umbral `count >= 2`.

## Resultados

### Clasificación de sentimiento

| Métrica | Valor |
|---|---|
| Accuracy en entrenamiento | 95.67% |
| **Accuracy en validación** | **66.00%** |
| Loss inicial → final | 0.7410 → 0.1072 |
| Épocas | 10 (Adam, `lr=1e-3`) |

Sin regularización el modelo llegaba a 98.3% / 58%. Agregar **dropout `p=0.1`** sobre los
pesos de atención y sobre la capa oculta del FFN —las dos posiciones que usa el paper
original— subió la validación 8 puntos y cerró la brecha de 40 a 30 puntos.

### Análisis de atención

La entropía media de la fila del `[CLS]` es **0.843 nats** contra **2.383** de una
distribución uniforme: la atención está fuertemente concentrada, típicamente en uno o dos
tokens.

Un hallazgo que matiza la lectura habitual de los mapas de atención: el token dominante
es frecuentemente el **sustantivo núcleo**, no la palabra que carga el sentimiento. En
`this movie is maddening` la atención va a `movie` con 0.774 y `maddening` no aparece en
el top. Lo que se agrega no son los tokens sino sus *value vectors*, así que los pesos de
atención no deben leerse como importancia de features.

### Generación y temperatura

Medido sobre 160 generaciones por temperatura:

| Temperatura | Fracción de `<UNK>` | Tokens generados (promedio) |
|---|---|---|
| 0.5 | 67.5% | 3.2 |
| 0.8 | 44.7% | 4.5 |
| 1.5 | 12.7% | 6.4 |

La temperatura **baja** produce **más** tokens desconocidos, no menos. `<UNK>` representa
el 28.5% del corpus de entrenamiento, así que es el token más probable en casi cualquier
contexto — es el argmax — y bajar la temperatura colapsa justo sobre el argmax.

Esto no es un defecto del modelo. Verificado por teacher forcing sobre el corpus:

| | |
|---|---|
| P(`<UNK>`) media que predice el modelo | 28.6% |
| Frecuencia real de `<UNK>` como siguiente token | 24.4% |
| `<UNK>` generado a temperatura 1.0 | 30.9% |
| Entropía media de la predicción | 2.79 nats (uniforme: 6.07) |

El modelo está bien calibrado: aprendió fielmente la distribución que se le dio. Lo que
está sesgado es esa distribución, porque el umbral `count >= 2` del tokenizador colapsa
1,133 tipos de palabra —el 72.6% del vocabulario— en un solo símbolo. El cuello de
botella está en el preprocesamiento y la escala de datos, no en la arquitectura.

## Correr el notebook

```bash
pip install torch matplotlib numpy
jupyter notebook S6_Proyecto1.ipynb
```

Ejecutar de arriba a abajo con **Restart & Run All**. El notebook descarga SST-2
automáticamente y las celdas de verificación validan cada parte contra los umbrales
esperados.

## Correr el artifact

El componente es React puro, sin dependencias externas:

```bash
npm create vite@latest transformer-viz -- --template react
cd transformer-viz && npm install
# reemplazar src/App.jsx con el contenido de TransformerViz.jsx
# vaciar src/index.css (los estilos por defecto de Vite descuadran el layout)
npm run dev
```

El archivo pesa ~512 KB porque lleva los pesos entrenados incrustados, así que la primera
compilación tarda un poco más de lo normal.

**Panel 1 — Sentimiento.** Recibe una oración, muestra la predicción con su confianza y
los mapas de atención de ambas cabezas.

**Panel 2 — Generación.** Recibe un texto semilla y un slider de temperatura entre 0.5 y
1.5, y muestra los mapas de atención causal. El vocabulario son 432 palabras; cualquier
palabra fuera de él se convierte en `<UNK>` dentro de la semilla misma.

Toda la matemática está reimplementada en JavaScript —multiplicación de matrices,
softmax, ReLU, LayerNorm, codificación posicional, atención multi-cabeza, máscaras— sin
llamar a ningún modelo externo.

## Autores

- Nelson Escalante
- Nicolle Gordillo

## Licencia

MIT
