# src/data/load_and_clean.py
"""
Pipeline de limpeza e pré-processamento de dados - VERSÃO FINAL APRIMORADA
Autor: Júlia Valandro Bonzanini
Projeto: Modelo Preditivo de Qualidade do Ar - Porto Alegre
"""

import pandas as pd
import numpy as np
import pytz
import os
import glob
import logging
from datetime import datetime
from pathlib import Path
import warnings
import json
import re  # Necessário para manipulação robusta de strings

warnings.filterwarnings('ignore')


# Configurações
class Config:
    TIMEZONE = 'America/Sao_Paulo'
    DATA_RAW_PATH = Path('../../data/raw')
    DATA_PROCESSED_PATH = Path('../../data/processed')
    REPORTS_PATH = Path('../../reports')

    # Período de interesse
    START_DATE = '2020-01-01'
    END_DATE = '2024-12-31'


# Configurar logging
def setup_logging():
    Config.REPORTS_PATH.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.REPORTS_PATH / 'data_pipeline.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.data = {}
        self._create_directories()

    def _create_directories(self):
        self.config.DATA_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
        logger.info("Diretórios criados/verificados")

    def run_pipeline(self):
        logger.info("INICIANDO PIPELINE DE PROCESSAMENTO DE DADOS")
        start_time = datetime.now()

        try:
            # Extrair dados
            self._extract_data()

            # Transformar dados
            self._transform_data()

            # Unificar dados
            unified_data = self._unify_data()

            if unified_data.empty:
                logger.error("Nenhum dado foi unificado")
                return None

            # Aplicar filtro temporal (2020-2024)
            filtered_data = self._filter_by_date_range(unified_data)

            # Engenharia de features
            final_data = self._feature_engineering(filtered_data)

            # Validar dados (VERSÃO DETALHADA)
            self._validate_data(final_data)

            # Salvar dados
            self._save_data(final_data)

            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Pipeline concluído em {execution_time:.2f} segundos")

            return final_data

        except Exception as e:
            logger.error(f"Erro no pipeline: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _extract_data(self):
        logger.info("Fase 1: Extração de dados")

        self._extract_fepam_data()
        self._extract_inmet_data()
        self._extract_datasus_data()
        self._extract_denatran_data()
        self._extract_inpe_data()

    def _extract_fepam_data(self):
        """Extrai dados FEPAM - converte XLSX para CSV primeiro se necessário"""
        logger.info("Processando dados FEPAM...")

        # Primeiro verificar se já existem CSVs convertidos
        fepam_csv_files = list(self.config.DATA_RAW_PATH.glob('*fepam*.csv'))
        fepam_xlsx_files = list(self.config.DATA_RAW_PATH.glob('*fepam*.xlsx'))

        all_data = []

        # Processar CSVs primeiro (se existirem)
        for file_path in fepam_csv_files:
            try:
                # Pular arquivos temporários se ainda existirem
                if file_path.name.startswith('temp_'):
                    continue

                logger.info(f"  Lendo CSV: {file_path.name}")
                df = self._read_fepam_csv(file_path)
                if not df.empty:
                    all_data.append(df)
                    logger.info(f"  OK {file_path.name}: {len(df)} registros")

            except Exception as e:
                logger.error(f"  ERRO em {file_path.name}: {e}")

        # Processar XLSX
        if fepam_xlsx_files:
            if not all_data:
                logger.info("  Convertendo arquivos XLSX para CSV...")

            for xlsx_file in fepam_xlsx_files:
                try:
                    csv_path = self._convert_xlsx_to_csv(xlsx_file)
                    if csv_path:
                        df = self._read_fepam_csv(csv_path)
                        if not df.empty:
                            all_data.append(df)
                            logger.info(f"  OK {xlsx_file.name}: {len(df)} registros")
                except Exception as e:
                    logger.error(f"  ERRO convertendo {xlsx_file.name}: {e}")

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            # Garantir que datetime seja único por estação
            combined = combined.sort_values('datetime').drop_duplicates(['datetime', 'estacao'])
            self.data['fepam'] = combined
            logger.info(f"Dados FEPAM combinados: {len(combined)} registros")
        else:
            logger.warning("Nenhum dado FEPAM processado")
            self.data['fepam'] = pd.DataFrame()

    def _convert_xlsx_to_csv(self, xlsx_path):
        """Converte arquivo XLSX para CSV temporário"""
        try:
            # Tentar ler com openpyxl se disponível
            try:
                # skiprows=1 para pular a linha de metadados antes do cabeçalho
                df = pd.read_excel(xlsx_path, skiprows=1)
            except ImportError:
                logger.error("Biblioteca openpyxl não instalada. Execute: pip install openpyxl")
                return None

            # Criar arquivo CSV temporário
            csv_path = self.config.DATA_RAW_PATH / f"temp_{xlsx_path.stem}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8')
            return csv_path

        except Exception as e:
            logger.error(f"Erro convertendo {xlsx_path.name}: {e}")
            return None

    def _read_fepam_csv(self, file_path):
        """Lê e processa arquivo CSV da FEPAM"""
        try:
            df = pd.read_csv(file_path, encoding='utf-8')

            # Verificar se a primeira coluna é data
            if len(df.columns) == 0:
                return pd.DataFrame()

            # A primeira coluna deve ser a data
            date_col = df.columns[0]

            # Renomear colunas baseado nos padrões observados
            column_mapping = {date_col: 'datetime'}
            polluant_mapping = {
                'pm10': 'PM10',
                'pm2,5': 'PM2_5',
                'pm2.5': 'PM2_5',
                'so2': 'SO2',
                'no2': 'NO2',
                'o3': 'O3',
                'co': 'CO'
            }

            for col in df.columns[1:]:  # Pular coluna de data
                col_lower = str(col).lower().strip()
                for key, value in polluant_mapping.items():
                    if key in col_lower:
                        column_mapping[col] = value
                        break

            df = df.rename(columns=column_mapping)

            # Processar datetime
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df = df.dropna(subset=['datetime'])
            df['datetime'] = df['datetime'].dt.tz_localize(self.config.TIMEZONE)

            # Converter colunas numéricas
            numeric_cols = ['PM10', 'PM2_5', 'SO2', 'NO2', 'O3', 'CO']
            for col in numeric_cols:
                if col in df.columns:
                    # Substituir vírgula por ponto e converter para numérico
                    df[col] = df[col].astype(str).str.replace(',', '.')
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Adicionar identificador da estação
            # 'CETE' no nome do arquivo indica Porto Alegre
            if 'Canoas' in file_path.name:
                df['estacao'] = 'Canoas'
            elif 'POA' in file_path.name or 'CETE' in file_path.name:
                df['estacao'] = 'Porto_Alegre'
            else:
                df['estacao'] = 'Desconhecida'

            # Manter apenas colunas relevantes
            cols_to_keep = ['datetime', 'estacao'] + [col for col in numeric_cols if col in df.columns]
            return df[cols_to_keep].copy()

        except Exception as e:
            logger.error(f"Erro lendo {file_path.name}: {e}")
            return pd.DataFrame()

    def _extract_inmet_data(self):
        """Extrai dados meteorológicos do INMET"""
        logger.info("Processando dados INMET...")

        inmet_files = list(self.config.DATA_RAW_PATH.glob('*inmet*.csv'))

        for file_path in inmet_files:
            try:
                logger.info(f"  Lendo {file_path.name}")

                # Ler metadados para saber quantas linhas pular
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # Encontrar linha que começa com "Data Medicao"
                skip_rows = 0
                for i, line in enumerate(lines):
                    if 'Data Medicao' in line:
                        skip_rows = i
                        break

                if skip_rows == 0:
                    skip_rows = 8  # Fallback

                # Ler dados
                df = pd.read_csv(file_path, sep=';', skiprows=skip_rows, encoding='utf-8', decimal=',')

                # Verificar colunas
                logger.info(f"    Colunas INMET: {df.columns.tolist()}")
                logger.info(f"    Número de colunas: {len(df.columns)}")

                # Renomear colunas
                col_names = ['data_medicao', 'hora_medicao', 'precipitacao', 'temperatura',
                             'temperatura_orvalho', 'umidade', 'vento_direcao', 'vento_velocidade', 'extra']

                # Manter apenas as colunas relevantes e renomear
                df = df.iloc[:, :8]
                df.columns = col_names[:8]

                # Combinar data e hora
                df['hora_medicao'] = df['hora_medicao'].astype(str).str.zfill(4)
                df['hora_medicao'] = df['hora_medicao'].str[:2] + ':' + df['hora_medicao'].str[2:4]
                df['datetime'] = pd.to_datetime(df['data_medicao'] + ' ' + df['hora_medicao'])
                df['datetime'] = df['datetime'].dt.tz_localize(self.config.TIMEZONE)

                # Converter colunas numéricas
                numeric_cols = ['precipitacao', 'temperatura', 'temperatura_orvalho',
                                'umidade', 'vento_direcao', 'vento_velocidade']
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                # Limpar e ordenar
                df_clean = df[['datetime'] + numeric_cols].copy()
                df_clean = df_clean.drop_duplicates('datetime').sort_values('datetime')

                self.data['inmet'] = df_clean
                logger.info(f"INMET: {len(df_clean)} registros")
                return

            except Exception as e:
                logger.error(f"Erro processando INMET: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue

        logger.warning("Nenhum dado INMET processado")
        self.data['inmet'] = pd.DataFrame()

    def _extract_datasus_data(self):
        """Extrai dados de saúde do DATASUS - Lógica de tratamento de meses robusta"""
        logger.info("Processando dados DATASUS...")

        try:
            datasus_files = list(self.config.DATA_RAW_PATH.glob('*datasus*.csv'))
            if not datasus_files:
                logger.warning("Nenhum arquivo DATASUS encontrado")
                self.data['datasus'] = pd.DataFrame()
                return

            file_path = datasus_files[0]
            logger.info(f"  Lendo {file_path.name}")

            # Ler pulando metadados - skiprows=5
            df = pd.read_csv(file_path, sep=';', skiprows=5, encoding='utf-8', on_bad_lines='skip', header=0)
            df.columns = [col.strip() for col in df.columns]

            # Encontrar colunas - Assumindo 1ª coluna é data e 2ª é internações
            date_col = None
            adm_col = None

            if len(df.columns) >= 2:
                date_col = df.columns[0]
                adm_col = df.columns[1]

            if date_col and adm_col:
                # Filtrar linhas com datas (que contém '/')
                mask = df[date_col].astype(str).str.contains('/', na=False)
                df_filtered = df[mask].copy()

                if df_filtered.empty:
                    logger.warning("Nenhuma linha com datas válidas encontrada no DATASUS")
                    self.data['datasus'] = pd.DataFrame()
                    return

                # Renomear para clareza
                df_filtered = df_filtered.rename(columns={date_col: 'mes_ano', adm_col: 'internacoes'})

                # Tratamento robusto de strings
                df_filtered['mes_ano'] = df_filtered['mes_ano'].astype(str).str.strip()
                df_filtered['mes_ano'] = df_filtered['mes_ano'].str.replace('..', '', regex=False).str.strip()

                # Mapeamento de meses em português (completo) para inglês abreviado
                month_replacements = {
                    'Janeiro': 'Jan', 'Fevereiro': 'Feb', 'Março': 'Mar', 'Abril': 'Apr',
                    'Maio': 'May', 'Junho': 'Jun', 'Julho': 'Jul', 'Agosto': 'Aug',
                    'Setembro': 'Sep', 'Outubro': 'Oct', 'Novembro': 'Nov', 'Dezembro': 'Dec'
                }

                for pt, en in month_replacements.items():
                    df_filtered['mes_ano'] = df_filtered['mes_ano'].str.replace(pt, en, case=False, regex=True)

                # Converter para datetime (assumindo dia 01 e formato %b)
                df_filtered['datetime'] = pd.to_datetime('01/' + df_filtered['mes_ano'],
                                                         format='%d/%b/%Y', errors='coerce')

                df_filtered = df_filtered.dropna(subset=['datetime'])

                # Filtro pelo período de interesse (2020-2024)
                start_date = pd.Timestamp(self.config.START_DATE)
                end_date = pd.Timestamp(self.config.END_DATE).replace(day=1)
                df_filtered = df_filtered[
                    (df_filtered['datetime'] >= start_date) & (df_filtered['datetime'] <= end_date)].copy()

                if df_filtered.empty:
                    logger.warning("Nenhum registro DATASUS no período 2020-2024 após conversão de data")
                    self.data['datasus'] = pd.DataFrame()
                    return

                df_filtered['datetime'] = df_filtered['datetime'].dt.tz_localize(self.config.TIMEZONE)

                # Converter internações para numérico
                df_filtered['internacoes'] = pd.to_numeric(df_filtered['internacoes'].astype(str), errors='coerce')
                df_filtered = df_filtered.dropna(subset=['internacoes']).copy()
                df_filtered['internacoes'] = df_filtered['internacoes'].astype(int)

                # Expandir para diário - Atribuir a média diária do mês
                all_daily = []
                for _, row in df_filtered.iterrows():
                    month_start = row['datetime']
                    month_end = month_start + pd.offsets.MonthEnd(0, normalize=True)
                    date_range = pd.date_range(start=month_start, end=month_end, freq='D',
                                               tz=self.config.TIMEZONE)

                    daily_average = row['internacoes'] / len(date_range)

                    month_df = pd.DataFrame({
                        'datetime': date_range,
                        'internacoes_respiratorias': daily_average
                    })
                    all_daily.append(month_df)

                if all_daily:
                    daily_df = pd.concat(all_daily, ignore_index=True)
                    self.data['datasus'] = daily_df.drop_duplicates('datetime').sort_values('datetime')
                    logger.info(f"DATASUS: {len(self.data['datasus'])} registros diários (média mensal aplicada)")
                else:
                    logger.warning("Nenhum dado diário gerado do DATASUS")
                    self.data['datasus'] = pd.DataFrame()
            else:
                logger.warning("Colunas de data/internações não encontradas no DATASUS")
                self.data['datasus'] = pd.DataFrame()

        except Exception as e:
            logger.error(f"Erro processando DATASUS: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.data['datasus'] = pd.DataFrame()

    def _extract_denatran_data(self):
        """Extrai dados de frota veicular - apenas 2020-2024"""
        logger.info("Processando dados DENATRAN...")

        try:
            frota_files = list(self.config.DATA_RAW_PATH.glob('*denatran*.csv'))
            if not frota_files:
                logger.warning("Nenhum arquivo DENATRAN encontrado")
                self.data['denatran'] = pd.DataFrame()
                return

            file_path = frota_files[0]
            logger.info(f"  Lendo {file_path.name}")

            df = pd.read_csv(file_path, sep=';', encoding='utf-8')

            # A primeira linha contém os totais por ano
            total_row = df.iloc[0]

            # Extrair anos das colunas - APENAS 2020-2024
            year_data = []
            for col in df.columns:
                # Garantir que o nome da coluna é um ano válido
                if str(col).strip().isdigit() and 2020 <= int(str(col).strip()) <= 2024:  # FILTRO APLICADO
                    try:
                        # Remover separadores e converter
                        valor = str(total_row[col]).replace('.', '').replace(',', '.')
                        year_data.append({'ano': int(str(col).strip()), 'frota_total': float(valor)})
                    except:
                        continue

            if not year_data:
                logger.warning("Nenhum dado anual encontrado para 2020-2024")
                self.data['denatran'] = pd.DataFrame()
                return

            frota_df = pd.DataFrame(year_data)
            logger.info(f"  Anos processados: {frota_df['ano'].tolist()}")

            # Expandir para diário
            all_daily = []
            for _, row in frota_df.iterrows():
                year = int(row['ano'])
                try:
                    # Criar datas sem timezone primeiro
                    start_date = pd.Timestamp(f'{year}-01-01')
                    end_date = pd.Timestamp(f'{year}-12-31')

                    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

                    # Aplicar timezone
                    date_range = date_range.tz_localize(self.config.TIMEZONE)

                    year_df = pd.DataFrame({
                        'datetime': date_range,
                        'frota_veicular': row['frota_total']
                    })
                    all_daily.append(year_df)
                except Exception as e:
                    logger.error(f"Erro processando ano {year}: {e}")
                    continue

            if all_daily:
                daily_df = pd.concat(all_daily, ignore_index=True)
                self.data['denatran'] = daily_df.drop_duplicates('datetime').sort_values('datetime')
                logger.info(f"DENATRAN: {len(self.data['denatran'])} registros diários (2020-2024)")
            else:
                self.data['denatran'] = pd.DataFrame()

        except Exception as e:
            logger.error(f"Erro processando DENATRAN: {e}")
            self.data['denatran'] = pd.DataFrame()

    def _extract_inpe_data(self):
        """Extrai dados de queimadas do INPE - preenche com zero"""
        logger.info("Processando dados INPE...")

        queimadas_files = list(self.config.DATA_RAW_PATH.glob('*inpe*.csv'))
        all_data = []

        for file_path in queimadas_files:
            try:
                logger.info(f"  Lendo {file_path.name}")

                df = pd.read_csv(file_path, sep=',', encoding='utf-8')

                # Converter data/hora
                df['datetime'] = pd.to_datetime(df['DataHora'], errors='coerce')
                df = df.dropna(subset=['datetime'])
                df['datetime'] = df['datetime'].dt.tz_localize(self.config.TIMEZONE)

                # Converter FRP para numérico
                df['FRP'] = pd.to_numeric(df['FRP'], errors='coerce')

                # Manter apenas colunas relevantes
                df_clean = df[['datetime', 'FRP']].copy()
                all_data.append(df_clean)
                logger.info(f"  OK {file_path.name}: {len(df_clean)} focos")

            except Exception as e:
                logger.error(f"  ERRO processando {file_path.name}: {e}")

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)

            # Agregar por dia
            combined['date'] = combined['datetime'].dt.date
            daily_agg = combined.groupby('date').agg({
                'FRP': ['count', 'sum', 'mean', 'max']
            }).reset_index()

            # Ajustar nomes das colunas
            daily_agg.columns = ['date', 'focos_queimadas_count', 'focos_queimadas_frp_sum',
                                 'focos_queimadas_frp_mean', 'focos_queimadas_frp_max']

            # Converter date para datetime com timezone (meia-noite)
            daily_agg['datetime'] = pd.to_datetime(daily_agg['date']).dt.tz_localize(self.config.TIMEZONE)
            daily_agg = daily_agg.drop('date', axis=1)

            # Criar índice completo de datas e preencher com zero
            start_date = pd.Timestamp(self.config.START_DATE).tz_localize(self.config.TIMEZONE)
            end_date = pd.Timestamp(self.config.END_DATE).tz_localize(self.config.TIMEZONE)
            full_date_range = pd.date_range(start=start_date, end=end_date, freq='D', tz=self.config.TIMEZONE)

            # Criar DataFrame completo com zeros
            full_df = pd.DataFrame({'datetime': full_date_range})
            full_df = full_df.set_index('datetime')
            full_df['focos_queimadas_count'] = 0.0
            full_df['focos_queimadas_frp_sum'] = 0.0
            full_df['focos_queimadas_frp_mean'] = 0.0
            full_df['focos_queimadas_frp_max'] = 0.0

            # Atualizar com dados reais onde existem
            if not daily_agg.empty:
                daily_agg = daily_agg.set_index('datetime')
                # Fazer o merge no índice
                full_df.update(daily_agg)
                # Garantir que o 'count' continue sendo um número inteiro
                full_df['focos_queimadas_count'] = full_df['focos_queimadas_count'].astype(int)

            # Manter 'datetime' como COLUNA para a próxima fase
            self.data['inpe'] = full_df.reset_index()
            logger.info(f"Queimadas: {len(self.data['inpe'])} dias (zeros preenchidos)")
        else:
            logger.warning("Nenhum dado de queimadas processado")
            # Criar DataFrame vazio com estrutura correta
            self.data['inpe'] = pd.DataFrame()

    def _transform_data(self):
        """Aplica transformações necessárias"""
        logger.info("Fase 2: Transformação de dados")

        for name, df in self.data.items():
            if not df.empty:
                try:
                    # Lógica limpa para INPE
                    if name == 'inpe' and 'datetime' in df.columns:
                        self.data[name] = df.set_index('datetime').sort_index()
                        logger.info(f"OK {name}: índice datetime aplicado (já agregado)")
                        continue

                    if 'datetime' in df.columns:
                        df = df.set_index('datetime')

                    # Resample para diário
                    if name == 'fepam':
                        # Para FEPAM, processar cada estação separadamente
                        if 'estacao' in df.columns:
                            stations = df['estacao'].unique()
                            resampled_dfs = []
                            for station in stations:
                                station_df = df[df['estacao'] == station].drop('estacao', axis=1, errors='ignore')
                                # Usar .mean() para agregação horária para diária
                                station_resampled = station_df.resample('D').mean()
                                station_resampled['estacao'] = station
                                resampled_dfs.append(station_resampled)

                            self.data[name] = pd.concat(resampled_dfs)
                        else:
                            self.data[name] = df.resample('D').mean()

                    elif name == 'datasus':
                        # DATASUS já foi calculado como média diária
                        self.data[name] = df.resample('D').mean()

                    else:
                        # Para outros dados (INMET, DENATRAN), usar mean
                        self.data[name] = df.resample('D').mean()

                    logger.info(f"OK {name}: resample aplicado")

                except Exception as e:
                    logger.error(f"ERRO transformando {name}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

    def _unify_data(self):
        """Unifica todos os datasets"""
        logger.info("Fase 3: Unificação de dados")

        # Começar com dados FEPAM
        base_df = self.data.get('fepam', pd.DataFrame())

        if base_df.empty:
            # Fallback (caso FEPAM esteja vazio, usar INMET como base temporal)
            if not self.data.get('inmet', pd.DataFrame()).empty:
                logger.warning("FEPAM vazio. Usando INMET como base temporal.")
                base_df = self.data.get('inmet')
            else:
                logger.error("Nenhum dado base disponível para unificação")
                return pd.DataFrame()

        # Para FEPAM com múltiplas estações, criar colunas separadas
        if 'estacao' in base_df.columns:
            # Fazer pivot para ter colunas por estação
            base_df_reset = base_df.reset_index()
            base_df_pivot = base_df_reset.pivot_table(
                index='datetime',
                columns='estacao'
            )
            # Achatar nomes das colunas
            base_df_pivot.columns = [f'{col[0]}_{col[1]}' for col in base_df_pivot.columns]
            base_df = base_df_pivot

        # Juntar outros datasets
        for name, df in self.data.items():
            if name != 'fepam' and not df.empty:
                try:
                    # Garantir que o index do dataframe a ser mergeado é datetime
                    if 'datetime' in df.columns:
                        df = df.set_index('datetime')
                    # Usar 'outer' para manter todas as datas
                    base_df = base_df.merge(df, left_index=True, right_index=True, how='outer')
                    logger.info(f"OK {name}: unificado com sucesso")
                except Exception as e:
                    logger.warning(f"ERRO unificando {name}: {e}")

        # Ordenar por data
        base_df = base_df.sort_index()

        logger.info(f"Dataset unificado: {len(base_df)} registros, {len(base_df.columns)} features")
        return base_df

    def _filter_by_date_range(self, df):
        """Filtra dados para o período 2020-2024"""
        logger.info("Aplicando filtro temporal: 2020-2024")

        start_date = pd.Timestamp(self.config.START_DATE).tz_localize(self.config.TIMEZONE)
        end_date = pd.Timestamp(self.config.END_DATE).tz_localize(self.config.TIMEZONE)

        # Usar loc para garantir que o índice é usado
        filtered_df = df.loc[(df.index >= start_date) & (df.index <= end_date)]

        logger.info(f"Dataset após filtro: {len(filtered_df)} registros (2020-2024)")
        return filtered_df

    def _feature_engineering(self, df):
        """Engenharia de features básica"""
        logger.info("Fase 4: Engenharia de Features")

        if df.empty:
            return df

        # Features temporais
        df['dayofyear'] = df.index.dayofyear
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['dayofweek'] = df.index.dayofweek
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        df['year'] = df.index.year

        # Features de sazonalidade
        df['sin_dayofyear'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)  # Usar 365.25 para anos bissextos
        df['cos_dayofyear'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)

        # Features de lag para PM2.5 (se existir)
        pm25_cols = [col for col in df.columns if 'PM2_5' in col]

        if pm25_cols:
            # Priorizar Porto Alegre, senão a primeira que aparecer
            target_col = next((col for col in pm25_cols if 'Porto_Alegre' in col), pm25_cols[0])
            logger.info(f"  Criando lags para coluna: {target_col}")

            for lag in [1, 2, 3, 7]:
                df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        else:
            logger.warning("  Nenhuma coluna PM2_5 encontrada para engenharia de lags.")

        logger.info(f"Features criadas: {len(df.columns)} colunas totais")
        return df

    def _validate_data(self, df):
        """Valida qualidade dos dados - VERSÃO DETALHADA CORRIGIDA"""
        logger.info("Fase 5: Validação de Dados")

        if df.empty:
            logger.error("Dataset final está vazio")
            return

        # Estatísticas básicas
        logger.info(f"Dataset final: {len(df)} linhas, {len(df.columns)} colunas")
        logger.info(f"Período: {df.index.min()} a {df.index.max()}")

        # Verificar quais fontes de dados estão presentes
        sources_present = []

        # 1. FEPAM (Verificar poluentes)
        fepam_cols = [col for col in df.columns if
                      any(pollutant in col for pollutant in ['PM10', 'PM2_5', 'SO2', 'NO2', 'O3', 'CO'])]
        if fepam_cols:
            sources_present.append('FEPAM')
            logger.info(f"  FEPAM: {len(fepam_cols)} colunas de poluentes")

        # 2. INMET (Verificar variáveis meteorológicas)
        inmet_check_cols = ['precipitacao', 'temperatura', 'umidade', 'vento_velocidade']
        inmet_cols = [col for col in df.columns if any(c in col for c in inmet_check_cols)]
        if inmet_cols:
            sources_present.append('INMET')
            logger.info(f"  INMET: {len(inmet_cols)} colunas de meteo")

        # 3. DATASUS
        if 'internacoes_respiratorias' in df.columns and df['internacoes_respiratorias'].notna().any():
            sources_present.append('DATASUS')

        # 4. DENATRAN
        if 'frota_veicular' in df.columns and df['frota_veicular'].notna().any():
            sources_present.append('DENATRAN')

        # 5. INPE
        if any('queimadas' in col for col in df.columns) and 'focos_queimadas_count' in df.columns and df[
            'focos_queimadas_count'].sum() > 0:
            sources_present.append('INPE')

        logger.info(f"Fontes de dados presentes: {sources_present}")

        # Verificar missing values por fonte
        logger.info("Missing values por fonte:")

        # FEPAM (Incluindo colunas de lag para PM2.5, que são baseadas em FEPAM)
        all_fepam_cols = [col for col in df.columns if
                          'PM' in col or 'CO' in col or 'NO2' in col or 'SO2' in col or 'O3' in col]

        if all_fepam_cols:
            fepam_missing = (df[all_fepam_cols].isnull().sum() / len(df)) * 100
            high_missing_fepam = fepam_missing[fepam_missing > 50]
            logger.info(f"  FEPAM (+ Lags) - Colunas com >50% missing: {len(high_missing_fepam)}")
            for col, pct in high_missing_fepam.items():
                logger.warning(f"    {col}: {pct:.1f}% missing")

        # INMET
        if inmet_cols:
            inmet_missing = (df[inmet_cols].isnull().sum() / len(df)) * 100
            high_missing_inmet = inmet_missing[inmet_missing > 50]
            logger.info(f"  INMET - Colunas com >50% missing: {len(high_missing_inmet)}")
            for col, pct in high_missing_inmet.items():
                logger.warning(f"    {col}: {pct:.1f}% missing")

        # DATASUS - CORREÇÃO CRÍTICA: Remover .iloc[0]
        if 'internacoes_respiratorias' in df.columns:
            datasus_col = 'internacoes_respiratorias'
            datasus_missing = (df[datasus_col].isnull().sum() / len(df)) * 100

            # datasus_missing é um float, não precisa de .iloc[0]
            if datasus_missing > 50:
                logger.warning(f"  DATASUS - Coluna com >50% missing: {datasus_col}: {datasus_missing:.1f}% missing")

        logger.info("Validação concluída")

    def _save_data(self, df):
        """Salva dados processados"""
        logger.info("Fase 6: Persistência de Dados")

        if df.empty:
            logger.error("Nenhum dado para salvar")
            return

        # CSV completo
        csv_path = self.config.DATA_PROCESSED_PATH / 'air_quality_processed.csv'
        df.to_csv(csv_path, index=True, encoding='utf-8')

        # Estatísticas do dataset - converter numpy para Python nativo
        stats = {
            'processing_date': datetime.now().isoformat(),
            'total_records': len(df),
            'total_features': len(df.columns),
            'date_range_start': df.index.min().isoformat() if not df.empty else None,
            'date_range_end': df.index.max().isoformat() if not df.empty else None,
            'data_sources_used': [name for name, data in self.data.items() if not data.empty],
            'columns_with_high_missing': (df.isnull().sum() / len(df) > 0.5).sum()
        }

        # Salvar estatísticas
        stats_path = self.config.DATA_PROCESSED_PATH / 'dataset_statistics.json'

        # Função para converter numpy types para Python nativo
        def convert_to_native(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            else:
                return obj

        stats_native = convert_to_native(stats)

        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats_native, f, indent=2, ensure_ascii=False)

        logger.info(f"Dados salvos:")
        logger.info(f"   CSV completo: {csv_path}")
        logger.info(f"   Estatísticas: {stats_path}")
        logger.info(f"   Período: {df.index.min().date()} a {df.index.max().date()}")
        logger.info(f"   Dimensões: {len(df)} linhas x {len(df.columns)} colunas")


def main():
    """Função principal"""
    try:
        config = Config()
        processor = DataProcessor(config)
        final_data = processor.run_pipeline()

        if final_data is not None:
            logger.info("Pipeline executado com sucesso!")
            print(f"\nRESUMO FINAL:")
            print(f"   Registros: {len(final_data)}")
            print(f"   Features: {len(final_data.columns)}")
            print(f"   Período: {final_data.index.min().date()} a {final_data.index.max().date()}")
            print(f"   Arquivos salvos em: {config.DATA_PROCESSED_PATH}")

            # Verificar fontes de dados presentes
            sources = []
            if any('PM' in col for col in final_data.columns):
                sources.append('FEPAM')
            if any(col in final_data.columns for col in ['precipitacao', 'temperatura', 'umidade']):
                sources.append('INMET')
            if 'internacoes_respiratorias' in final_data.columns and final_data[
                'internacoes_respiratorias'].notna().any():
                sources.append('DATASUS')
            if 'frota_veicular' in final_data.columns and final_data['frota_veicular'].notna().any():
                sources.append('DENATRAN')
            if any('queimadas' in col for col in final_data.columns):
                sources.append('INPE')

            print(f"   Fontes de dados incluídas: {sources}")
        else:
            logger.error("Pipeline falhou!")

        return final_data

    except Exception as e:
        logger.error(f"Erro crítico: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    final_df = main()