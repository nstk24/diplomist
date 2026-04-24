# Raman Spectra Local Web App

Локальное веб-приложение на `Streamlit` для переноса анализа из ноутбука в нормальную структуру:

- слой загрузки данных и preprocessing;
- статистический анализ SNR;
- PCA, t-SNE и UMAP;
- сравнение классификаторов;
- nested CV для более честной оценки процедуры выбора модели;
- интерпретация коэффициентов логистической регрессии.

## Структура

- `app.py` — точка входа веб-приложения.
- `raman_webapp/data.py` — загрузка Excel/CSV и сборка рабочего датасета.
- `raman_webapp/preprocessing.py` — ALS baseline correction, SNV, SNR.
- `raman_webapp/analysis.py` — статистика и проекции.
- `raman_webapp/modeling.py` — screening, nested CV, интерпретация модели.
- `raman_webapp/visuals.py` — графики для UI.

## Запуск

1. Создать и активировать виртуальное окружение.
2. Установить зависимости:

```powershell
pip install -r requirements.txt
```

3. Запустить приложение:

```powershell
streamlit run app.py
```

После запуска приложение откроется на локальном адресе вида `http://localhost:8501`.

## Источники данных

Приложение загружает и пересчитывает preprocessing только из `Raman_krov_SSZ-zdorovye.xlsx`.
Использование предвычисленных CSV для обучения отключено.

## Замечание

В текущем shell-окружении этого репозитория Python-пакеты для scientific stack не установлены, поэтому здесь была выполнена синтаксическая проверка кода, но не полный runtime smoke test приложения. Для реального запуска нужен Python с установленными пакетами из `requirements.txt`.
