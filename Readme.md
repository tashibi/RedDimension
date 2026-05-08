# RedDimension: Transformer-Based 2D to 3D Generative Engine 🔴📐

**RedDimension** — это легковесный, но мощный движок для реконструкции 3D-геометрии из одиночных изображений. Система использует архитектуру **SDF-Transformer**, оптимизированную для работы на видеокартах **AMD** через стек **ROCm**.

## 🚀 Ключевые особенности
- **Transformer Decoder:** 12-слойная архитектура с механизмом Cross-Attention для точной генерации поверхностей.
- **AMD Optimized:** Полная поддержка RDNA 3/4 (GFX 12.0.1) и ROCm 7.x.
- **SDF Representation:** Генерация гладких мешей с использованием функций знакового расстояния.
- **Inference & Training:** Полный цикл от подготовки датасета до финального GUI.

## 🛠 Технический стек
- **Backend:** PyTorch (ROCm)
- **Geometry:** Custom Transformer + Marching Cubes
- **Optimization:** 8-bit Adam (bitsandbytes) + Mixed Precision (AMP)
- **UI:** Gradio

## 📁 Структура проекта
- `model.py` — Ядро нейросети (Encoder-Transformer).
- `train_image_to_3d.py` — Скрипт обучения с поддержкой 8-bit оптимизаторов.
- `step2_convert_to_sdf.py` — Подготовка 3D данных в формат SDF.
- `step5_render_dataset.py` — Рендерер датасета (10 ракурсов на модель).
- `app.py` — Пользовательский интерфейс генерации.

## 💻 Установка и запуск

1. **Установите зависимости:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Обучение:**
   Поместите ваши модели в `dataset/raw/` и последовательно запустите `step2`, `step5`, а затем `train_image_to_3d.py`.

3. **Генерация:**
   Поместите обученные веса в папку `weights/` и запустите:
   ```bash
   python app.py
   ```

## 📄 Лицензия
MIT
