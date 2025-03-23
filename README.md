import logging
import requests
import time
from datetime import datetime, time as dt_time
import pytz
import pandas as pd
import os
import configparser
import matplotlib.pyplot as plt
import json
from openpyxl.drawing import image
from sklearn.linear_model import LinearRegression
import random
import numpy as np
from scipy.signal import find_peaks

# --- ML imports ---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

def load_config(config_file='config.ini'):
    try:
        config = configparser.ConfigParser()
        config.read(config_file)
        return config
    except FileNotFoundError:
        logging.error(f"Файл конфигурации {config_file} не найден.")
        print(f"Ошибка: Файл конфигурации {config_file} не найден. \nРешение: Убедитесь, что файл '{config_file}' существует и находится в правильной директории.")
        return None
    except configparser.Error as e:
        logging.error(f"Ошибка при чтении файла конфигурации: {e}")
        print(f"Ошибка: Ошибка при чтении файла конфигурации: {e}. \nРешение: Проверьте синтаксис файла '{config_file}'. Возможно, есть опечатки или неправильные значения.")
        return None

def setup_logging(config):
    if not config:
        print("Предупреждение: Объект конфигурации отсутствует. Логирование не будет настроено.")
        return
    try:
        log_level_str = config.get('Logging', 'level', fallback='INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        print(f"Ошибка: Проблема с настройками логирования в файле конфигурации: {e}. \nРешение: Проверьте секцию 'Logging' и опцию 'level' в файле конфигурации.")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s') #default logging
    except Exception as e:
        print(f"Ошибка при настройке логирования: {e}. \nРешение: Проверьте общие настройки логирования. Возможно, есть конфликт с другими библиотеками.")
        logging.error(f"Ошибка при настройке логирования: {e}")

# --- data_loader.py (Загрузка данных) ---
def fetch_data(api_key, symbol, api_function='TIME_SERIES_DAILY', retry_attempts=3, retry_delay=5):
    if not api_key or not symbol:
        logging.error("Необходимо предоставить API ключ и символ для запроса данных.")
        print("Ошибка: Необходимо предоставить API ключ и символ для запроса данных. \nРешение: Убедитесь, что API ключ и символ переданы в функцию.")
        return None
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': api_function,
        'symbol': symbol,
        'apikey': api_key
    }
    for attempt in range(retry_attempts):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            logging.info(f"Получены данные для {symbol} с ключом {api_key} (попытка {attempt + 1})")
            return data
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP Error для {symbol} (попытка {attempt + 1}): {e}")
            print(
                f"Ошибка HTTP: {e}. \nРешение: Проверьте правильность API ключа и символа. Возможно, превышен лимит запросов к API.")
            time.sleep(retry_delay)
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error для {symbol} (попытка {attempt + 1}): {e}")
            print(
                f"Ошибка запроса: {e}. \nРешение: Проверьте подключение к интернету. Возможно, API временно недоступен.")
            time.sleep(retry_delay)
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error для {symbol} (попытка {attempt + 1}): {e}")
            print(
                f"Ошибка декодирования JSON: {e}. \nРешение: Возможно, API вернул данные в неправильном формате. Попробуйте повторить запрос позже.")
            time.sleep(retry_delay)
        except Exception as e:
            logging.error(f"Непредвиденная ошибка для {symbol} (попытка {attempt + 1}): {e}")
            print(
                f"Непредвиденная ошибка: {e}. \nРешение: Проверьте логи для получения дополнительной информации. Возможно, есть проблема в коде или с API.")
            time.sleep(retry_delay)
    logging.error(f"Не удалось получить данные для {symbol} после {retry_attempts} попыток")
    print(
        f"Ошибка: Не удалось получить данные для {symbol} после {retry_attempts} попыток. \nРешение: Проверьте API ключ, символ и подключение к интернету. Увеличьте количество попыток или задержку между ними.")
    return None

def load_local_data(symbol, data_dir='data'):
    if not symbol:
        logging.error("Необходимо предоставить символ для загрузки локальных данных.")
        print("Ошибка: Необходимо предоставить символ для загрузки локальных данных. \nРешение: Убедитесь, что символ передан в функцию.")
        return None
    file_path = os.path.join(data_dir, f"{symbol}_{datetime.now().strftime('%Y%m%d')}.json")
    if not os.path.exists(file_path):
        logging.warning(f"Локальный файл {file_path} не найден.")
        print(f"Предупреждение: Локальный файл {file_path} не найден. \nРешение: Убедитесь, что файл существует и находится в указанной директории. Если файла нет, попробуйте загрузить данные из API.")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Данные для {symbol} загружены из локального файла {file_path}")
        return data
    except FileNotFoundError:
        logging.error(f"Файл {file_path} не найден.")
        print(f"Ошибка: Файл {file_path} не найден. \nРешение: Убедитесь, что файл существует и находится в указанной директории.")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Ошибка декодирования JSON из файла {file_path}: {e}")
        print(f"Ошибка: Ошибка декодирования JSON из файла {file_path}: {e}. \nРешение: Проверьте формат файла. Возможно, он поврежден или содержит неправильный JSON.")
        return None
    except Exception as e:
        logging.error(f"Ошибка при загрузке данных из файла {file_path}: {e}")
        print(f"Ошибка: Ошибка при загрузке данных из файла {file_path}: {e}. \nРешение: Проверьте права доступа к файлу и директории. Возможно, файл занят другим процессом.")
        return None

def get_stock_data(symbol, config):
    if not config:
        logging.error("Необходимо предоставить объект конфигурации.")
        print("Ошибка: Необходимо предоставить объект конфигурации. \nРешение: Убедитесь, что объект конфигурации передан в функцию.")
        return None
    api_keys = config.get('API', 'api_keys', fallback='').split(',')
    api_function = config.get('API', 'function', fallback='TIME_SERIES_DAILY')
    data_dir = config.get('Files', 'data_directory', fallback='data')
    retry_attempts = config.getint('API', 'retry_attempts', fallback=3)
    retry_delay = config.getint('API', 'retry_delay', fallback=5)

    if not api_keys:
        logging.warning("Список API ключей пуст. Будет предпринята попытка загрузки только из локального файла.")
        print("Предупреждение: Список API ключей пуст. Будет предпринята попытка загрузки только из локального файла. \nРешение: Добавьте API ключи в файл конфигурации.")

    data = load_local_data(symbol, data_dir)
    if data:
        if isinstance(data, dict) and 'Time Series (Daily)' in data:
            return data
        else:
            logging.warning(f"Некорректный формат данных в локальном файле для {symbol}.  Попытка загрузки из API.")
            print(f"Предупреждение: Некорректный формат данных в локальном файле для {symbol}. Будет предпринята попытка загрузки из API. \nРешение: Проверьте формат локального файла. Возможно, он был поврежден или создан неправильно.")
            data = None

    if data is None:
        for api_key in api_keys:
            data = fetch_data(api_key, symbol, api_function, retry_attempts, retry_delay)
            if data and 'Time Series (Daily)' in data:
                if not os.path.exists(data_dir):
                    try:
                        os.makedirs(data_dir)
                    except OSError as e:
                        logging.error(f"Не удалось создать директорию {data_dir}: {e}")
                        print(f"Ошибка: Не удалось создать директорию {data_dir}: {e}. \nРешение: Проверьте права доступа к директории. Возможно, директория уже существует.")
                        return None
                file_path = os.path.join(data_dir, f"{symbol}_{datetime.now().strftime('%Y%m%d')}.json")
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=4)
                except OSError as e:
                    logging.error(f"Не удалось сохранить данные в файл {file_path}: {e}")
                    print(f"Ошибка: Не удалось сохранить данные в файл {file_path}: {e}. \nРешение: Проверьте права доступа к файлу. Возможно, файл занят другим процессом или директория не существует.")
                    return None
                return data
            else:
                logging.warning(f"Не удалось получить данные для {symbol} с ключом {api_key}.")
                print(f"Предупреждение: Не удалось получить данные для {symbol} с ключом {api_key}. Будет предпринята попытка с другим API ключом, если он есть. \nРешение: Проверьте работоспособность API ключа. Возможно, он заблокирован или превышен лимит запросов.")
            time.sleep(random.uniform(30, 40))
    return None

def create_dataframe(data):
    if not data:
        logging.error("Данные для создания DataFrame отсутствуют.")
        print("Ошибка: Данные для создания DataFrame отсутствуют. \nРешение: Убедитесь, что данные были успешно загружены из API или локального файла.")
        return None
    if 'Time Series (Daily)' not in data:
        logging.error("Неверный формат данных: ключ 'Time Series (Daily)' не найден.")
        print("Ошибка: Неверный формат данных: ключ 'Time Series (Daily)' не найден. \nРешение: Убедитесь, что API возвращает данные в ожидаемом формате.")
        return None
    try:
        df = pd.DataFrame(data['Time Series (Daily)']).T
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df
    except KeyError as e:
        logging.error(f"Отсутствует ключ в данных: {e}")
        print(f"Ошибка: Отсутствует ключ в данных: {e}. \nРешение: Проверьте структуру данных, возвращаемых API. Возможно, структура изменилась.")
        return None
    except ValueError as e:
        logging.error(f"Ошибка при преобразовании типов данных: {e}")
        print(f"Ошибка: Ошибка при преобразовании типов данных: {e}. \nРешение: Проверьте, что данные, возвращаемые API, имеют правильный формат (числа).")
        return None
    except Exception as e:
        logging.error(f"Ошибка при создании DataFrame: {e}")
        print(f"Ошибка при создании DataFrame: {e}. \nРешение: Проверьте общие настройки pandas и структуру данных.")
        return None

def calculate_stochastic_oscillator(df, window):
    if df is None or df.empty:
        print("Предупреждение: DataFrame пуст или отсутствует. Расчет стохастического осциллятора невозможен.")
        return df
    try:
        df['Lowest_14'] = df['low'].rolling(window=window).min()
        df['Highest_14'] = df['high'].rolling(window=window).max()
        df['Stoch'] = 100 * ((df['close'] - df['Lowest_14']) / (df['Highest_14'] - df['Lowest_14']))
        return df
    except KeyError as e:
        logging.error(f"Отсутствует столбец в DataFrame: {e}")
        print(f"Ошибка: Отсутствует столбец в DataFrame: {e}. \nРешение: Убедитесь, что в DataFrame есть столбцы 'low', 'high', 'close'.")
        return df
    except Exception as e:
        logging.error(f"Ошибка при расчете стохастического осциллятора: {e}")
        print(f"Ошибка при расчете стохастического осциллятора: {e}. \nРешение: Проверьте общие настройки pandas и DataFrame.")
        return df

def calculate_ichimoku_cloud(df):
    if df is None or df.empty:
        print("Предупреждение: DataFrame пуст или отсутствует. Расчет облака Ишимоку невозможен.")
        return df
    try:
        nine_period_high = df['high'].rolling(window=9).max()
        nine_period_low = df['low'].rolling(window=9).min()
        df['Tenkan_sen'] = (nine_period_high + nine_period_low) / 2
        twenty_six_period_high = df['high'].rolling(window=26).max()
        twenty_six_period_low = df['low'].rolling(window=26).min()
        df['Kijun_sen'] = (twenty_six_period_high + twenty_six_period_low) / 2
        df['Senkou_Span_A'] = (df['Tenkan_sen'] + df['Kijun_sen']) / 2
        fifty_two_period_high = df['high'].rolling(window=52).max()
        fifty_two_period_low = df['low'].rolling(window=52).min()
        df['Senkou_Span_B'] = (fifty_two_period_high + fifty_two_period_low) / 2
        df['Chikou_Span'] = df['close'].shift(-26)
        return df
    except KeyError as e:
        logging.error(f"Отсутствует столбец в DataFrame: {e}")
        print(f"Ошибка: Отсутствует столбец в DataFrame: {e}. \nРешение: Убедитесь, что в DataFrame есть столбцы 'low', 'high', 'close'.")
        return df
    except Exception as e:
        logging.error(f"Ошибка при расчете облака Ишимоку: {e}")
        print(f"Ошибка при расчете облака Ишимоку: {e}. \nРешение: Проверьте общие настройки pandas и DataFrame.")
        return df

def calculate_sma(df, window):
    if df is None or df.empty:
        print("Предупреждение: DataFrame пуст или отсутствует. Расчет SMA невозможен.")
        return df
    try:
        df['SMA'] = df['close'].rolling(window=window).mean()
        return df
    except KeyError as e:
        logging.error(f"Отсутствует столбец в DataFrame: {e}")
        print(f"Ошибка: Отсутствует столбец в DataFrame: {e}. \nРешение: Убедитесь, что в DataFrame есть столбец 'close'.")
        return df
    except Exception as e:
        logging.error(f"Ошибка при расчете SMA: {e}")
        print(f"Ошибка при расчете SMA: {e}. \nРешение: Проверьте общие настройки pandas и DataFrame.")
        return df

def calculate_ema(df, window):
    if df is None or df.empty:
        print("Предупреждение: DataFrame пуст или отсутствует. Расчет EMA невозможен.")
        return df
    try:
        df['EMA'] = df['close'].ewm(span=window, adjust=False).mean()
        return df
    except KeyError as e:
        logging.error(f"Отсутствует столбец в DataFrame: {e}")
        print(f"Ошибка: Отсутствует столбец в DataFrame: {e}. \nРешение: Убедитесь, что в DataFrame есть столбец 'close'.")
        return df
    except Exception as e:
        logging.error(f"Ошибка при расчете EMA: {e}")
        print(f"Ошибка при расчете EMA: {e}. \nРешение: Проверьте общие настройки pandas и DataFrame.")
        return df
def calculate_rsi(df, window):
    if df is None or df.empty:
        print("Предупреждение: DataFrame пуст или отсутствует. Расчет RSI невозможен.")
        return df

    try:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss

        # Handle division by zero
        if (loss == 0).any():
            print("Предупреждение: Обнаружено деление на ноль при расчете RSI. Заменяем RSI на 50 в этих точках.")
            rs[loss == 0] = 1  # Avoid division by zero

        df['RSI'] = 100 - (100 / (1 + rs))
        return df

    except KeyError as e:
        logging.error(f"Отсутствует столбец в DataFrame: {e}")
        print(f"Ошибка: Отсутствует столбец в DataFrame: {e}. \nРешение: Убедитесь, что в DataFrame есть столбец 'close'.")
        return df
    except Exception as e:
        logging.error(f"Ошибка при расчете RSI: {e}")
        print(f"Ошибка при расчете RSI: {e}. \nРешение: Проверьте общие настройки pandas и DataFrame.")
        return df

def calculate_macd(df, short_window, long_window, signal_window):
    """Рассчитывает MACD."""
    df['EMA_short'] = df['close'].ewm(span=short_window, adjust=False).mean()
    df['EMA_long'] = df['close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = df['EMA_short'] - df['EMA_long']
    df['Signal'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    return df

def calculate_bollinger_bands(df, window, num_std):
    """Рассчитывает полосы Боллинджера."""
    df['SMA'] = df['close'].rolling(window=window).mean()
    df['STD'] = df['close'].rolling(window=window).std()
    df['UpperBB'] = df['SMA'] + (df['STD'] * num_std)
    df['LowerBB'] = df['SMA'] - (df['STD'] * num_std)
    return df

def calculate_volume_indicators(df, volume_sma_window):
    """Рассчитывает индикаторы объема."""
    df['Volume_SMA'] = df['volume'].rolling(window=volume_sma_window).mean()
    return df

def calculate_all_indicators(df, config):
    """Рассчитывает все технические индикаторы."""
    sma_window = config.getint('Indicators', 'sma_window', fallback=20)
    ema_window = config.getint('Indicators', 'ema_window', fallback=20)
    rsi_window = config.getint('Indicators', 'rsi_window', fallback=14)
    macd_short_window = config.getint('Indicators', 'macd_short_ema', fallback=12)
    macd_long_window = config.getint('Indicators', 'macd_long_ema', fallback=26)
    macd_signal_window = config.getint('Indicators', 'macd_signal_ema', fallback=9)
    bb_window = config.getint('Indicators', 'bb_window', fallback=20)
    bb_std = config.getfloat('Indicators', 'bb_std', fallback=2)
    volume_sma_window = config.getint('Indicators', 'volume_sma_window', fallback=20)
    stochastic_window = config.getint('Indicators', 'stochastic_window', fallback=14)

    df = calculate_sma(df, sma_window)
    df = calculate_ema(df, ema_window)
    df = calculate_rsi(df, rsi_window)
    df = calculate_macd(df, macd_short_window, macd_long_window, macd_signal_window)
    df = calculate_bollinger_bands(df, bb_window, bb_std)
    df = calculate_volume_indicators(df, volume_sma_window)
    df = calculate_stochastic_oscillator(df, stochastic_window)
    df = calculate_ichimoku_cloud(df)

    return df

# --- trading_strategy.py (Торговые стратегии) ---
def detect_pattern(df):
    """Обнаруживает простые графические паттерны (например, голова и плечи)."""
    window = 20
    if df.shape[0] >= window * 3:
        try:
            left_shoulder = df['high'].iloc[-window*3:-window*2].max()
            head = df['high'].iloc[-window*2:-window].max()
            right_shoulder = df['high'].iloc[-window:].max()

            neckline1 = df['low'].iloc[-window*3:-window*2].min()
            neckline2 = df['low'].iloc[-window*2:-window].min()

            if head > left_shoulder and head > right_shoulder and abs(left_shoulder - right_shoulder) < 0.1 * head and abs(neckline1 - neckline2) < 0.1 * head:
                return "head_and_shoulders"
        except (IndexError, KeyError):
            return None

    return None

def analyze_wave(df, min_peak_distance=10):
    """Выполняет волновой анализ и формирует простые торговые сигналы."""
    peaks, _ = find_peaks(df['close'].values, distance=min_peak_distance)
    valleys, _ = find_peaks(-df['close'].values, distance=min_peak_distance)

    peak_coords = [(i, df['close'].iloc[i]) for i in peaks]
    valley_coords = [(i, df['close'].iloc[i]) for i in valleys]

    if not peak_coords or not valley_coords:
        return "Недостаточно данных для волнового анализа", None

    last_peak_idx, last_peak_value = peak_coords[-1]
    last_valley_idx, last_valley_value = valley_coords[-1]

    if last_peak_idx > last_valley_idx:
        return "Возможен нисходящий тренд (волновой анализ)", 'sell'
    else:
        return "Возможен восходящий тренд (волновой анализ)", 'buy'

def predict_trend(df, days=7):
    """Прогнозирует направление тренда на основе линейной регрессии."""
    if len(df) < 30:
        return None

    try:
        X = np.arange(len(df) - 30, len(df)).reshape(-1, 1)
        y = df['close'].values[-30:]
        model = LinearRegression()
        model.fit(X, y)

        future_X = np.array([len(df) + days - 1]).reshape(-1, 1)
        predicted_price = model.predict(future_X)[0]

        current_price = df['close'].iloc[-1]
        trend = "восходящий" if predicted_price > current_price else "нисходящий"
        return predicted_price, trend

    except Exception as e:
        logging.error(f"Ошибка при прогнозировании тренда: {e}")
        return None

def calculate_profit(current_price, predicted_price, holding_period):
    """Рассчитывает потенциальную прибыль и период удержания."""
    try:
        profit_percentage = ((predicted_price - current_price) / current_price) * 100
        return holding_period, profit_percentage
    except (ValueError, ZeroDivisionError) as e:
        logging.warning(f"Ошибка при расчете прибыли: {e}")
        return None, None

# --- ML functions ---
def prepare_ml_data(df, config):
    """Подготовка данных для машинного обучения."""
    df = calculate_all_indicators(df, config)

    # Create target variable (example logic)
    df['Target'] = 0  # Default: hold
    df.loc[df['RSI'] < config.getint('Trading', 'rsi_oversold', fallback=30), 'Target'] = 1  # Buy
    df.loc[df['RSI'] > 70, 'Target'] = -1  # Sell

    df = df.dropna()

    # Select features
    features = ['close', 'SMA', 'RSI', 'MACD', 'Volume_SMA', 'Stoch', 'Tenkan_sen', 'Kijun_sen', 'Senkou_Span_A', 'Senkou_Span_B']
    X = df[features]
    y = df['Target']

    return X, y

def train_ml_model(X, y, filename="stock_model.pkl"):
    """Обучение модели машинного обучения."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность модели: {accuracy}")

    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Модель сохранена в {filename}")
    return model

def load_ml_model(filename="stock_model.pkl"):
    """Загрузка обученной модели машинного обучения."""
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Модель загружена из {filename}")
        return model
    except FileNotFoundError:
        print(f"Файл {filename} не найден.  Требуется обучение модели.")
        return None

def make_trade_decision_ml(df, config, model):
    """Принятие торгового решения с использованием машинного обучения."""
    df = calculate_all_indicators(df, config)  # Ensure indicators are calculated
    df = df.dropna()  # Drop NaN values after calculating indicators

    features = ['close', 'SMA', 'RSI', 'MACD', 'Volume_SMA', 'Stoch', 'Tenkan_sen', 'Kijun_sen', 'Senkou_Span_A', 'Senkou_Span_B']
    if not all(feature in df.columns for feature in features):
        print("Не хватает данных для ML модели.")
        return "Недостаточно данных для ML", None

    latest_data = df[features].iloc[[-1]]

    prediction = model.predict(latest_data)[0]

    if prediction == 1:
        decision = "Купить (ML)"
        action = 'buy'
    elif prediction == -1:
        decision = "Продать (ML)"
        action = 'sell'
    else:
        decision = "Держать (ML)"
        action = None

    return decision, action

def make_trade_decision(df, config, ml_model=None):
    """Принимает решение о торговле на основе индикаторов и ML."""
    # (Mostly unchanged from original, but added ml_model parameter)
    sma_window = config.getint('Indicators', 'sma_window', fallback=20)
    rsi_oversold = config.getint('Trading', 'rsi_oversold', fallback=30)
    trend_prediction_days = config.getint('Trading', 'trend_prediction_days', fallback=7)
    profit_threshold = config.getfloat('Trading', 'profit_threshold', fallback=5.0)
    volume_threshold = config.getfloat('Trading', 'volume_threshold', fallback=1.5)
    indicators_to_use = config.get('Trading', 'indicators_to_use', fallback='SMA,RSI,Volume').split(',')
    enable_adaptive_strategy = config.getboolean('Trading', 'enable_adaptive_strategy', fallback=True)

    if df.shape[0] < sma_window:
        return "Недостаточно данных для анализа", None, None

    latest_data = df.iloc[-1]
    decision = "Нет четких сигналов для торговли"
    action = None
    hold_period = None
    profit_percentage = None

    pattern = detect_pattern(df)
    if pattern:
        if "head_and_shoulders" in pattern:
            decision = "Продать (Head and Shoulders)"
            action = 'sell'
        elif "double_bottom" in pattern:
            decision = "Купить (Double Bottom)"
            action = 'buy'

    wave_analysis_result, wave_action = analyze_wave(df)
    if wave_analysis_result:
        decision = wave_analysis_result
        if wave_action:
            action = wave_action

    base_rules = []
    if 'SMA' in indicators_to_use and latest_data['close'] > latest_data['SMA']:
        base_rules.append("Цена выше SMA")
    if 'RSI' in indicators_to_use and latest_data['RSI'] < rsi_oversold:
        base_rules.append("RSI перепродан")
    if 'Volume' in indicators_to_use and latest_data['volume'] > latest_data['Volume_SMA'] * volume_threshold:
        base_rules.append("Высокий объем")

    if base_rules and action is None:
        if len(base_rules) >= 2:
            decision = f"{'Купить' if 'RSI перепродан' in base_rules else 'Продать'} (Индикаторы)"
            action = 'buy' if 'RSI перепродан' in base_rules else 'sell'

    if action:
        trend_info = predict_trend(df, days=trend_prediction_days)
        if trend_info:
            predicted_price, trend = trend_info
            hold_period, profit_percentage = calculate_profit(latest_data['close'], predicted_price, trend_prediction_days)

            if profit_percentage is not None and abs(profit_percentage) >= profit_threshold:
                decision = f"{'Купить' if action == 'buy' else 'Продать'} (Прогноз: {trend}, период: {hold_period} дн., прибыль: {profit_percentage:.2f}%)"
            else:
                decision = "Нет четкого сигнала (низкая прибыль)"
                action = None

    if enable_adaptive_strategy and decision == "Нет четких сигналов для торговли":
        original_sma_window = config.getint('Indicators', 'sma_window')

        for test_sma in range(original_sma_window - 10, original_sma_window + 11, 5):
            if test_sma > 0:
                config['Indicators']['sma_window'] = str(test_sma)
                df = calculate_all_indicators(df, config)

                temp_decision, temp_hold_period, temp_profit_percentage = make_trade_decision(df, config)

                if temp_decision != "Нет четких сигналов для торговли":
                    decision = temp_decision
                    action = None
                    hold_period = temp_hold_period
                    profit_percentage = temp_profit_percentage
                    print(f"Адаптивная стратегия: SMA окно установлено в {test_sma}")
                    break

        config['Indicators']['sma_window'] = str(original_sma_window)
        df = calculate_all_indicators(df, config)

    # Add ML decision:
    if ml_model:
        ml_decision, ml_action = make_trade_decision_ml(df, config, ml_model)
        if ml_action:  # If the ML model gives a signal
            decision = ml_decision  # Override the decision
            action = ml_action
            print(f"ML Model Signal: {decision}")


    if action is None:
        decision = "Нет четких сигналов для торговли"
        hold_period = None
        profit_percentage = None

    return decision, hold_period, profit_percentage

# --- report_generator.py (Генерация отчетов) ---
def generate_excel_report(all_data, error_data, output_path):
    """Генерирует отчет в формате Excel."""
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            if all_data:
                for item in all_data:
                    symbol = item['symbol']
                    df = item['data']
                    decision = item['decision']
                    api_key_used = item.get('api_key_used', 'Неизвестно')
                    hold_period = item.get('hold_period', None)
                    profit_percentage = item.get('profit_percentage', None)

                    df.to_excel(writer, sheet_name=symbol, index=True)
                    worksheet = writer.sheets[symbol]

                    worksheet.cell(row=1, column=1).value = "Тикер"
                    worksheet.cell(row=1, column=2).value = symbol
                    worksheet.cell(row=2, column=1).value = "Период"
                    worksheet.cell(row=2, column=2).value = f"{df.index[0].strftime('%Y-%m-%d')} - {df.index[-1].strftime('%Y-%m-%d')}"
                    worksheet.cell(row=3, column=1).value = "Дата анализа"
                    worksheet.cell(row=3, column=2).value = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    worksheet.cell(row=4, column=1).value = "Решение"
                    worksheet.cell(row=4, column=2).value = decision
                    worksheet.cell(row=5, column=1).value = "Использованный API ключ"
                    worksheet.cell(row=5, column=2).value = api_key_used
                    if hold_period is not None:
                        worksheet.cell(row=6, column=1).value = "Период удержания (дни)"
                        worksheet.cell(row=6, column=2).value = hold_period
                    if profit_percentage is not None:
                         worksheet.cell(row=7, column=1).value = "Потенциальная прибыль (%)"
                         worksheet.cell(row=7, column=2).value = f"{profit_percentage:.2f}"

                    try:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(df.index, df['close'], label='Цена закрытия', linewidth=1)
                        ax.plot(df.index, df['SMA'], label='SMA', linewidth=1)
                        if 'UpperBB' in df.columns and 'LowerBB' in df.columns:
                            ax.plot(df.index, df['UpperBB'], label='Upper BB', linestyle='--', linewidth=0.7)
                            ax.plot(df.index, df['LowerBB'], label='Lower BB', linestyle='--', linewidth=0.7)
                        ax.set_title(f'Анализ {symbol}')
                        ax.set_xlabel('Дата')
                        ax.set_ylabel('Цена')
                        ax.legend()
                        fig.tight_layout()

                        image_path = os.path.join(os.getcwd(), f"{symbol}_chart.png")
                        fig.savefig(image_path)
                        plt.close(fig)
                        img = image.Image(image_path)
                        img.anchor = 'H1'
                        worksheet.add_image(img)
                        os.remove(image_path)

                    except Exception as e:
                        logging.error(f"Error generating chart for {symbol}: {e}")

                print(f"Анализ завершён. Результаты сохранены в '{output_path}'.")
            else:
                print("Нет доступных данных для сохранения.")

            if error_data:
                error_df = pd.DataFrame(error_data)
                error_df.to_excel(writer, sheet_name='Ошибки', index=False)
                print("Ошибки записаны в 'stock_analysis.xlsx' на листе 'Ошибки'.")

    except Exception as e:
        logging.error(f"Ошибка при сохранении в Excel: {e}")
        print(f"Критическая ошибка при сохранении в Excel: {e}")

# --- main.py (Основной модуль) ---
def is_market_open():
    """Проверяет, открыта ли Московская биржа."""
    tz = pytz.timezone("Europe/Moscow")
    now = datetime.now(tz)
    if now.weekday() < 7:
        market_open = dt_time(9, 0)
        market_close = dt_time(22, 0)
        return market_open <= now.time() <= market_close
    return False

def main():
    print('Запуск проекта "AVGUST", всем приготовится')

    config = load_config()
    setup_logging(config)

    if not is_market_open():
        logging.warning("Биржа закрыта!")
        print("Биржа закрыта. Анализ невозможен.")
        return

    symbols = [s.strip() for s in config.get('Symbols', 'symbols', fallback='').split(',')]
    output_path = config.get('Files', 'output_path', fallback='stock_analysis.xlsx')
    model_filename = "stock_model.pkl"

    all_data = []
    error_data = []

    # Train or load the ML model *ONCE* before the loop
    first_symbol_data = get_stock_data(symbols[0], config)  # Get data for one symbol
    if not first_symbol_data:
        print(f"Не удалось получить данные для {symbols[0]}.  Невозможно обучить модель.")
        return

    first_symbol_df = create_dataframe(first_symbol_data)
    if first_symbol_df is None:
        print(f"Не удалось создать DataFrame для {symbols[0]}. Невозможно обучить модель.")
        return

    X, y = prepare_ml_data(first_symbol_df.copy(), config) # IMPORTANT: Use .copy()
    ml_model = load_ml_model(model_filename) # Try to load
    if ml_model is None: # If not loaded, train
        ml_model = train_ml_model(X, y, model_filename) # Train the model

    for symbol in symbols:
        logging.info(f"Анализ символа: {symbol}")
        api_key_used = None

        data = get_stock_data(symbol, config)
        if not data:
            error_data.append({'symbol': symbol, 'error': f'Не удалось получить данные для {symbol}'})
            continue

        df = create_dataframe(data)
        if df is None:
            error_data.append({'symbol': symbol, 'error': f'Ошибка при создании DataFrame для {symbol}'})
            continue

        df = calculate_all_indicators(df, config)

        # Use the make_trade_decision function, passing the ML model
        decision, hold_period, profit_percentage = make_trade_decision(df, config, ml_model)
        logging.info(f"Решение по {symbol}: {decision}")
        print(f"Решение по {symbol}: {decision}")


        all_data.append({
            'symbol': symbol,
            'decision': decision,
            'data': df,
            'api_key_used': api_key_used,
            'hold_period': hold_period,
            'profit_percentage': profit_percentage
        })

        time.sleep(random.uniform(30, 40))

    generate_excel_report(all_data, error_data, output_path)

if __name__ == "__main__":
    main()
