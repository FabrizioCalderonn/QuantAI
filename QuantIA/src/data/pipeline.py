"""
Pipeline de datos para el sistema de trading cuantitativo.
Maneja la ingesta, limpieza, validación y almacenamiento de datos de mercado.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import hashlib
import json

from ..utils.config import load_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataPipeline:
    """
    Pipeline principal para la gestión de datos de mercado.
    
    Responsabilidades:
    - Ingesta de datos de múltiples fuentes
    - Limpieza y validación de datos
    - Sincronización de time zones
    - Almacenamiento eficiente
    - Control de calidad
    """
    
    def __init__(self, config_path: str = "configs/default_parameters.yaml"):
        """
        Inicializa el pipeline de datos.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config = load_config(config_path)
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Crear directorios si no existen
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuración de instrumentos
        self.instruments = self.config['instrumentos']
        self.timeframes = self.config['timeframes']
        self.timezone = self.config['zona_horaria']
        
        # Cache para evitar descargas repetidas
        self.cache = {}
        
        logger.info("DataPipeline inicializado correctamente")
    
    def download_data(self, 
                     symbols: List[str], 
                     start_date: str, 
                     end_date: str,
                     interval: str = "1h") -> Dict[str, pd.DataFrame]:
        """
        Descarga datos históricos de múltiples fuentes.
        
        Args:
            symbols: Lista de símbolos a descargar
            start_date: Fecha de inicio (YYYY-MM-DD)
            end_date: Fecha de fin (YYYY-MM-DD)
            interval: Intervalo de tiempo (1m, 5m, 15m, 30m, 1h, 1d)
            
        Returns:
            Diccionario con DataFrames por símbolo
        """
        logger.info(f"Descargando datos para {len(symbols)} símbolos desde {start_date} hasta {end_date}")
        
        data = {}
        failed_downloads = []
        
        for symbol in symbols:
            try:
                logger.info(f"Descargando {symbol}...")
                
                # Verificar cache primero
                cache_key = f"{symbol}_{start_date}_{end_date}_{interval}"
                if cache_key in self.cache:
                    logger.info(f"Usando datos en cache para {symbol}")
                    data[symbol] = self.cache[cache_key]
                    continue
                
                # Descargar datos
                ticker_data = self._download_symbol(symbol, start_date, end_date, interval)
                
                if ticker_data is not None and not ticker_data.empty:
                    data[symbol] = ticker_data
                    self.cache[cache_key] = ticker_data
                    logger.info(f"Datos descargados para {symbol}: {len(ticker_data)} registros")
                else:
                    failed_downloads.append(symbol)
                    logger.warning(f"Falló la descarga de {symbol}")
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error descargando {symbol}: {str(e)}")
                failed_downloads.append(symbol)
        
        if failed_downloads:
            logger.warning(f"Fallos en descarga: {failed_downloads}")
        
        logger.info(f"Descarga completada: {len(data)} símbolos exitosos, {len(failed_downloads)} fallos")
        return data
    
    def _download_symbol(self, 
                        symbol: str, 
                        start_date: str, 
                        end_date: str,
                        interval: str) -> Optional[pd.DataFrame]:
        """
        Descarga datos para un símbolo específico.
        
        Args:
            symbol: Símbolo del instrumento
            start_date: Fecha de inicio
            end_date: Fecha de fin
            interval: Intervalo de tiempo
            
        Returns:
            DataFrame con datos OHLCV o None si falla
        """
        try:
            # Usar yfinance como fuente principal
            ticker = yf.Ticker(symbol)
            
            # Mapear intervalos de yfinance
            interval_map = {
                "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
                "1h": "1h", "4h": "4h", "1d": "1d", "1wk": "1wk", "1mo": "1mo"
            }
            
            yf_interval = interval_map.get(interval, "1h")
            
            # Descargar datos
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=yf_interval,
                auto_adjust=True,
                prepost=True,
                threads=True
            )
            
            if data.empty:
                logger.warning(f"No se encontraron datos para {symbol}")
                return None
            
            # Limpiar y estandarizar columnas
            data = self._clean_raw_data(data, symbol)
            
            return data
            
        except Exception as e:
            logger.error(f"Error descargando {symbol}: {str(e)}")
            return None
    
    def _clean_raw_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Limpia datos crudos descargados.
        
        Args:
            data: DataFrame con datos crudos
            symbol: Símbolo del instrumento
            
        Returns:
            DataFrame limpio y estandarizado
        """
        # Crear copia para evitar modificaciones in-place
        clean_data = data.copy()
        
        # Estandarizar nombres de columnas
        column_mapping = {
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        }
        
        clean_data = clean_data.rename(columns=column_mapping)
        
        # Asegurar que tenemos las columnas necesarias
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in clean_data.columns]
        
        if missing_columns:
            logger.error(f"Columnas faltantes para {symbol}: {missing_columns}")
            return pd.DataFrame()
        
        # Convertir timezone a UTC
        if clean_data.index.tz is None:
            clean_data.index = clean_data.index.tz_localize('UTC')
        else:
            clean_data.index = clean_data.index.tz_convert('UTC')
        
        # Validar datos
        clean_data = self._validate_data(clean_data, symbol)
        
        # Agregar metadatos
        clean_data.attrs['symbol'] = symbol
        clean_data.attrs['downloaded_at'] = datetime.now().isoformat()
        clean_data.attrs['data_hash'] = self._calculate_data_hash(clean_data)
        
        return clean_data
    
    def _validate_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Valida la calidad de los datos.
        
        Args:
            data: DataFrame a validar
            symbol: Símbolo del instrumento
            
        Returns:
            DataFrame validado
        """
        original_length = len(data)
        
        # Remover filas con valores NaN en columnas críticas
        critical_columns = ['open', 'high', 'low', 'close']
        data = data.dropna(subset=critical_columns)
        
        # Validar lógica OHLC
        invalid_ohlc = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        
        if invalid_ohlc.any():
            logger.warning(f"Encontrados {invalid_ohlc.sum()} registros con OHLC inválido en {symbol}")
            data = data[~invalid_ohlc]
        
        # Remover outliers extremos (precios negativos o muy altos)
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            data = data[data[col] > 0]
            # Remover precios que sean > 10x la mediana (posibles errores)
            median_price = data[col].median()
            data = data[data[col] < median_price * 10]
        
        # Validar volumen
        if 'volume' in data.columns:
            data = data[data['volume'] >= 0]
        
        # Remover duplicados por timestamp
        data = data[~data.index.duplicated(keep='last')]
        
        # Ordenar por timestamp
        data = data.sort_index()
        
        removed_count = original_length - len(data)
        if removed_count > 0:
            logger.info(f"Removidos {removed_count} registros inválidos de {symbol}")
        
        return data
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """
        Calcula hash de los datos para verificar integridad.
        
        Args:
            data: DataFrame a hashear
            
        Returns:
            Hash MD5 de los datos
        """
        # Usar solo columnas numéricas para el hash
        numeric_data = data.select_dtypes(include=[np.number])
        data_string = numeric_data.to_string()
        return hashlib.md5(data_string.encode()).hexdigest()
    
    def save_data(self, data: Dict[str, pd.DataFrame], 
                  data_type: str = "raw") -> None:
        """
        Guarda datos en formato parquet.
        
        Args:
            data: Diccionario con DataFrames por símbolo
            data_type: Tipo de datos (raw, processed)
        """
        save_dir = self.raw_dir if data_type == "raw" else self.processed_dir
        
        for symbol, df in data.items():
            if df.empty:
                continue
                
            filename = f"{symbol}_{data_type}.parquet"
            filepath = save_dir / filename
            
            try:
                df.to_parquet(filepath, compression='snappy')
                logger.info(f"Datos guardados: {filepath}")
                
                # Guardar metadatos
                metadata = {
                    'symbol': symbol,
                    'data_type': data_type,
                    'saved_at': datetime.now().isoformat(),
                    'shape': df.shape,
                    'date_range': {
                        'start': df.index.min().isoformat(),
                        'end': df.index.max().isoformat()
                    },
                    'data_hash': df.attrs.get('data_hash', ''),
                    'columns': list(df.columns)
                }
                
                metadata_file = save_dir / f"{symbol}_{data_type}_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
            except Exception as e:
                logger.error(f"Error guardando {symbol}: {str(e)}")
    
    def load_data(self, symbol: str, data_type: str = "raw") -> Optional[pd.DataFrame]:
        """
        Carga datos desde archivos parquet.
        
        Args:
            symbol: Símbolo del instrumento
            data_type: Tipo de datos (raw, processed)
            
        Returns:
            DataFrame con datos o None si no existe
        """
        load_dir = self.raw_dir if data_type == "raw" else self.processed_dir
        filepath = load_dir / f"{symbol}_{data_type}.parquet"
        
        if not filepath.exists():
            logger.warning(f"Archivo no encontrado: {filepath}")
            return None
        
        try:
            data = pd.read_parquet(filepath)
            logger.info(f"Datos cargados: {filepath}")
            return data
        except Exception as e:
            logger.error(f"Error cargando {symbol}: {str(e)}")
            return None
    
    def get_data_manifest(self) -> pd.DataFrame:
        """
        Genera manifiesto de todos los datos disponibles.
        
        Returns:
            DataFrame con información de todos los archivos de datos
        """
        manifest_data = []
        
        for data_dir in [self.raw_dir, self.processed_dir]:
            data_type = "raw" if data_dir == self.raw_dir else "processed"
            
            for filepath in data_dir.glob("*.parquet"):
                symbol = filepath.stem.replace(f"_{data_type}", "")
                
                try:
                    # Cargar metadatos si existen
                    metadata_file = data_dir / f"{symbol}_{data_type}_metadata.json"
                    metadata = {}
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                    
                    # Información del archivo
                    file_info = {
                        'symbol': symbol,
                        'data_type': data_type,
                        'filepath': str(filepath),
                        'file_size_mb': filepath.stat().st_size / (1024 * 1024),
                        'modified_at': datetime.fromtimestamp(filepath.stat().st_mtime).isoformat(),
                        'shape': metadata.get('shape', ''),
                        'date_range_start': metadata.get('date_range', {}).get('start', ''),
                        'date_range_end': metadata.get('date_range', {}).get('end', ''),
                        'data_hash': metadata.get('data_hash', '')
                    }
                    
                    manifest_data.append(file_info)
                    
                except Exception as e:
                    logger.error(f"Error procesando {filepath}: {str(e)}")
        
        manifest_df = pd.DataFrame(manifest_data)
        
        if not manifest_df.empty:
            manifest_df = manifest_df.sort_values(['symbol', 'data_type'])
        
        return manifest_df
    
    def run_full_pipeline(self, 
                         start_date: str = None, 
                         end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Ejecuta el pipeline completo de datos.
        
        Args:
            start_date: Fecha de inicio (por defecto: 5 años atrás)
            end_date: Fecha de fin (por defecto: hoy)
            
        Returns:
            Diccionario con todos los datos procesados
        """
        # Configurar fechas por defecto
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            years_back = self.config.get('ventana_historica_anos', 5)
            start_date = (datetime.now() - timedelta(days=years_back * 365)).strftime('%Y-%m-%d')
        
        logger.info(f"Iniciando pipeline completo desde {start_date} hasta {end_date}")
        
        # Obtener todos los símbolos
        all_symbols = []
        for category, symbols in self.instruments.items():
            all_symbols.extend(symbols)
        
        # Descargar datos
        raw_data = self.download_data(
            symbols=all_symbols,
            start_date=start_date,
            end_date=end_date,
            interval=self.timeframes['principal']
        )
        
        # Guardar datos crudos
        self.save_data(raw_data, data_type="raw")
        
        # Procesar datos (limpieza adicional, feature engineering básico)
        processed_data = self._process_data(raw_data)
        
        # Guardar datos procesados
        self.save_data(processed_data, data_type="processed")
        
        # Generar manifiesto
        manifest = self.get_data_manifest()
        manifest.to_csv(self.data_dir / "data_manifest.csv", index=False)
        
        logger.info("Pipeline completo ejecutado exitosamente")
        return processed_data
    
    def _process_data(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Procesa datos crudos (limpieza adicional, cálculos básicos).
        
        Args:
            raw_data: Diccionario con datos crudos
            
        Returns:
            Diccionario con datos procesados
        """
        processed_data = {}
        
        for symbol, df in raw_data.items():
            if df.empty:
                continue
            
            processed_df = df.copy()
            
            # Calcular retornos
            processed_df['returns'] = processed_df['close'].pct_change()
            processed_df['log_returns'] = np.log(processed_df['close'] / processed_df['close'].shift(1))
            
            # Calcular volatilidad rolling
            processed_df['volatility_20'] = processed_df['returns'].rolling(20).std() * np.sqrt(252)
            processed_df['volatility_50'] = processed_df['returns'].rolling(50).std() * np.sqrt(252)
            
            # Calcular ATR
            processed_df['atr_14'] = self._calculate_atr(processed_df, 14)
            
            # Calcular indicadores básicos
            processed_df['sma_20'] = processed_df['close'].rolling(20).mean()
            processed_df['sma_50'] = processed_df['close'].rolling(50).mean()
            processed_df['rsi_14'] = self._calculate_rsi(processed_df['close'], 14)
            
            # Remover NaN iniciales
            processed_df = processed_df.dropna()
            
            processed_data[symbol] = processed_df
            
            logger.info(f"Datos procesados para {symbol}: {len(processed_df)} registros")
        
        return processed_data
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calcula Average True Range (ATR).
        
        Args:
            df: DataFrame con datos OHLC
            period: Período para ATR
            
        Returns:
            Serie con valores ATR
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calcula Relative Strength Index (RSI).
        
        Args:
            prices: Serie de precios
            period: Período para RSI
            
        Returns:
            Serie con valores RSI
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


def main():
    """
    Función principal para ejecutar el pipeline de datos.
    """
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Crear y ejecutar pipeline
    pipeline = DataPipeline()
    
    # Ejecutar pipeline completo
    data = pipeline.run_full_pipeline()
    
    # Mostrar resumen
    print(f"\nPipeline completado exitosamente!")
    print(f"Total de instrumentos procesados: {len(data)}")
    
    for symbol, df in data.items():
        print(f"{symbol}: {len(df)} registros desde {df.index.min()} hasta {df.index.max()}")


if __name__ == "__main__":
    main()

