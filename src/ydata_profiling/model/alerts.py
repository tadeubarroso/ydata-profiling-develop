"""Logic for alerting the user on possibly problematic patterns in the data (e.g. high number of zeros , constant
values, high correlations)."""

from enum import Enum, auto, unique
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from ydata_profiling.config import Settings
from ydata_profiling.model.correlations import perform_check_correlation
from ydata_profiling.utils.styles import get_alert_styles


def fmt_percent(value: float, edge_cases: bool = True) -> str:
    """Format a ratio as a percentage.

    Args:
        edge_cases: Check for edge cases?
        value: The ratio.

    Returns:
        The percentage with 1 point precision.
    """
    if edge_cases and round(value, 3) == 0 and value > 0:
        return "< 0.1%"
    if edge_cases and round(value, 3) == 1 and value < 1:
        return "> 99.9%"

    return f"{value*100:2.1f}%"


@unique
class AlertType(Enum):
    """Alert types"""

    CONSTANT = auto()
    """Esta variável tem um valor constante."""

    ZEROS = auto()
    """Esta variável contém zeros."""

    HIGH_CORRELATION = auto()
    """Esta variável está altamente correlacionada."""

    HIGH_CARDINALITY = auto()
    """Esta variável tem uma cardinalidade elevada."""

    UNSUPPORTED = auto()
    """Esta variável não é suportada."""

    DUPLICATES = auto()
    """Esta variável contém duplicados."""

    NEAR_DUPLICATES = auto()
    """Esta variável contém duplicados."""

    SKEWED = auto()
    """Esta variável está fortemente assimétrica."""

    IMBALANCE = auto()
    """Esta variável está desequilibrada."""

    MISSING = auto()
    """Esta variável contém valores em falta."""

    INFINITE = auto()
    """Esta variável contém valores infinitos."""

    TYPE_DATE = auto()
    """Esta variável é provavelmente do tipo data/hora, mas é tratada como categórica."""

    UNIQUE = auto()
    """Esta variável tem valores únicos."""

    DIRTY_CATEGORY = auto()
    """Esta variável é uma variável categórica com potenciais valores imprecisos e, por essa razão, pode causar problemas de consistência."""

    CONSTANT_LENGTH = auto()
    """Esta variável tem um comprimento constante."""

    REJECTED = auto()
    """As variáveis são rejeitadas se não as quisermos considerar para análise posterior."""

    UNIFORM = auto()
    """A variável está uniformemente distribuída."""

    NON_STATIONARY = auto()
    """A variável é uma série não estacionária."""

    SEASONAL = auto()
    """A variável é uma série temporal sazonal."""

    EMPTY = auto()
    """O DataFrame está vazio."""


class Alert:
    """An alert object (type, values, column)."""

    _anchor_id: Optional[str] = None

    def __init__(
        self,
        alert_type: AlertType,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        fields: Optional[Set] = None,
        is_empty: bool = False,
    ):
        self.fields = fields or set()
        self.alert_type = alert_type
        self.values = values or {}
        self.column_name = column_name
        self._is_empty = is_empty
        self._styles = get_alert_styles()

    @property
    def alert_type_name(self) -> str:
        return self.alert_type.name.replace("_", " ").capitalize()

    @property
    def anchor_id(self) -> Optional[str]:
        if self._anchor_id is None:
            self._anchor_id = str(hash(self.column_name))
        return self._anchor_id

    def fmt(self) -> str:
        # TODO: render in template
        style = self._styles.get(self.alert_type.name.lower(), "secondary")
        hint = ""

        if self.alert_type == AlertType.HIGH_CORRELATION and self.values is not None:
            num = len(self.values["fields"])
            title = ", ".join(self.values["fields"])
            corr = self.values["corr"]
            hint = f'data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="Esta variável tem uma elevada correlação {corr} com {num} campos: {title}"'

        return (
            f'<span class="badge text-bg-{style}" {hint}>{self.alert_type_name}</span>'
        )

    def _get_description(self) -> str:
        """Return a human level description of the alert.

        Returns:
            str: alert description
        """
        alert_type = self.alert_type.name
        column = self.column_name
        return f"[{alert_type}] alert on column {column}"

    def __repr__(self):
        return self._get_description()


class ConstantLengthAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.CONSTANT_LENGTH,
            values=values,
            column_name=column_name,
            fields={"composition_min_length", "composition_max_length"},
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        return f"[{self.column_name}] has a constant length"


class ConstantAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.CONSTANT,
            values=values,
            column_name=column_name,
            fields={"n_distinct"},
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        return f"[{self.column_name}] has a constant value"


class DuplicatesAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.DUPLICATES,
            values=values,
            column_name=column_name,
            fields={"n_duplicates"},
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        if self.values is not None:
            return f"O conjunto de dados tem {self.values['n_duplicates']} ({fmt_percent(self.values['p_duplicates'])}) linhas duplicadas."
        else:
            return "O conjunto de dados não tem linhas duplicadas."


class NearDuplicatesAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.NEAR_DUPLICATES,
            values=values,
            column_name=column_name,
            fields={"n_near_dups"},
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        if self.values is not None:
            return f"O conjunto de dados tem {self.values['n_near_dups']} ({fmt_percent(self.values['p_near_dups'])}) linhas quase idênticas"
        else:
            return "O conjunto de dados não tem linhas quase idênticas"


class EmptyAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.EMPTY,
            values=values,
            column_name=column_name,
            fields={"n"},
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        return "O conjunto de dados está vazio."


class HighCardinalityAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.HIGH_CARDINALITY,
            values=values,
            column_name=column_name,
            fields={"n_distinct"},
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        if self.values is not None:
            return f"[{self.column_name}] tem {self.values['n_distinct']:} ({fmt_percent(self.values['p_distinct'])}) valores distintos."
        else:
            return f"[{self.column_name}] tem uma alta cardinalidade"


class DirtyCategoryAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.DIRTY_CATEGORY,
            values=values,
            column_name=column_name,
            fields={"n_fuzzy_vals"},
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        if self.values is not None:
            return f"[{self.column_name}] has {self.values['n_fuzzy_vals']} fuzzy values: {fmt_percent(self.values['p_fuzzy_vals'])} por categoria."
        else:
            return f"[{self.column_name}] sem valores inválidos nas categorias."


class HighCorrelationAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.HIGH_CORRELATION,
            values=values,
            column_name=column_name,
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        if self.values is not None:
            description = f"[{self.column_name}] está altamente {self.values['corr']} ccorrelacionado(a) com [{self.values['fields'][0]}]"
            if len(self.values["fields"]) > 1:
                description += f" e {len(self.values['fields']) - 1} outros campos."
        else:
            return (
                f"[{self.column_name}] apresenta uma correlação elevada com uma ou mais colunas"
            )
        return description


class ImbalanceAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.IMBALANCE,
            values=values,
            column_name=column_name,
            fields={"imbalance"},
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        description = f"[{self.column_name}] está altamente desequilibrado"
        if self.values is not None:
            return description + f" ({self.values['imbalance']})"
        else:
            return description


class InfiniteAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.INFINITE,
            values=values,
            column_name=column_name,
            fields={"p_infinite", "n_infinite"},
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        if self.values is not None:
            return f"[{self.column_name}] tem {self.values['n_infinite']} ({fmt_percent(self.values['p_infinite'])}) valores infinitos"
        else:
            return f"[{self.column_name}] tem valores infinitos"


class MissingAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.MISSING,
            values=values,
            column_name=column_name,
            fields={"p_missing", "n_missing"},
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        if self.values is not None:
            return f"[{self.column_name}] {self.values['n_missing']} ({fmt_percent(self.values['p_missing'])}) valores ausentes"
        else:
            return f"[{self.column_name}] tem valores ausentes"


class NonStationaryAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.NON_STATIONARY,
            values=values,
            column_name=column_name,
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        return f"[{self.column_name}] é não estacionário"


class SeasonalAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.SEASONAL,
            values=values,
            column_name=column_name,
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        return f"[{self.column_name}] é sazonal"


class SkewedAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.SKEWED,
            values=values,
            column_name=column_name,
            fields={"skewness"},
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        description = f"[{self.column_name}] apresenta forte assimetria"
        if self.values is not None:
            return description + f"(\u03b31 = {self.values['skewness']})"
        else:
            return description


class TypeDateAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.TYPE_DATE,
            values=values,
            column_name=column_name,
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        return f"[{self.column_name}] contém apenas valores de data e hora, mas é categórico(a). Considere aplicar `pd.to_datetime()`"


class UniformAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.UNIFORM,
            values=values,
            column_name=column_name,
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        return f"[{self.column_name}] tem distribuição uniforme"


class UniqueAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.UNIQUE,
            values=values,
            column_name=column_name,
            fields={"n_distinct", "p_distinct", "n_unique", "p_unique"},
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        return f"[{self.column_name}] tem valores únicos"


class UnsupportedAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.UNSUPPORTED,
            values=values,
            column_name=column_name,
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        return f"[{self.column_name}] é um tipo não suportado, verifique se necessita de limpeza ou análise adicional"


class ZerosAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.ZEROS,
            values=values,
            column_name=column_name,
            fields={"n_zeros", "p_zeros"},
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        if self.values is not None:
            return f"[{self.column_name}] tem {self.values['n_zeros']} ({fmt_percent(self.values['p_zeros'])}) zeros"
        else:
            return f"[{self.column_name}] tem predominantemente zeros"


class RejectedAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.REJECTED,
            values=values,
            column_name=column_name,
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        return f"[{self.column_name}] foi rejeitado"


def check_table_alerts(table: dict) -> List[Alert]:
    """Checks the overall dataset for alerts.

    Args:
        table: Overall dataset statistics.

    Returns:
        A list of alerts.
    """
    alerts: List[Alert] = []
    if alert_value(table.get("n_duplicates", np.nan)):
        alerts.append(
            DuplicatesAlert(
                values=table,
            )
        )
    if table["n"] == 0:
        alerts.append(
            EmptyAlert(
                values=table,
            )
        )
    return alerts


def numeric_alerts(config: Settings, summary: dict) -> List[Alert]:
    alerts: List[Alert] = []

    # Skewness
    if skewness_alert(summary["skewness"], config.vars.num.skewness_threshold):
        alerts.append(SkewedAlert(summary))

    # Infinite values
    if alert_value(summary["p_infinite"]):
        alerts.append(InfiniteAlert(summary))

    # Zeros
    if alert_value(summary["p_zeros"]):
        alerts.append(ZerosAlert(summary))

    if (
        "chi_squared" in summary
        and summary["chi_squared"]["pvalue"] > config.vars.num.chi_squared_threshold
    ):
        alerts.append(UniformAlert())

    return alerts


def timeseries_alerts(config: Settings, summary: dict) -> List[Alert]:
    alerts: List[Alert] = numeric_alerts(config, summary)

    if not summary["stationary"]:
        alerts.append(NonStationaryAlert())

    if summary["seasonal"]:
        alerts.append(SeasonalAlert())

    return alerts


def categorical_alerts(config: Settings, summary: dict) -> List[Alert]:
    alerts: List[Alert] = []

    # High cardinality
    if summary.get("n_distinct", np.nan) > config.vars.cat.cardinality_threshold:
        alerts.append(HighCardinalityAlert(summary))

    if (
        "chi_squared" in summary
        and summary["chi_squared"]["pvalue"] > config.vars.cat.chi_squared_threshold
    ):
        alerts.append(UniformAlert())

    if summary.get("date_warning"):
        alerts.append(TypeDateAlert())

    # Constant length
    if "composition" in summary and summary["min_length"] == summary["max_length"]:
        alerts.append(ConstantLengthAlert())

    # Imbalance
    if (
        "imbalance" in summary
        and summary["imbalance"] > config.vars.cat.imbalance_threshold
    ):
        alerts.append(ImbalanceAlert(summary))
    return alerts


def boolean_alerts(config: Settings, summary: dict) -> List[Alert]:
    alerts: List[Alert] = []

    if (
        "imbalance" in summary
        and summary["imbalance"] > config.vars.bool.imbalance_threshold
    ):
        alerts.append(ImbalanceAlert())
    return alerts


def generic_alerts(summary: dict) -> List[Alert]:
    alerts: List[Alert] = []

    # Missing
    if alert_value(summary["p_missing"]):
        alerts.append(MissingAlert())

    return alerts


def supported_alerts(summary: dict) -> List[Alert]:
    alerts: List[Alert] = []

    if summary.get("n_distinct", np.nan) == summary["n"]:
        alerts.append(UniqueAlert())
    if summary.get("n_distinct", np.nan) == 1:
        alerts.append(ConstantAlert(summary))
    return alerts


def unsupported_alerts() -> List[Alert]:
    alerts: List[Alert] = [
        UnsupportedAlert(),
        RejectedAlert(),
    ]
    return alerts


def check_variable_alerts(config: Settings, col: str, description: dict) -> List[Alert]:
    """Checks individual variables for alerts.

    Args:
        col: The column name that is checked.
        description: The series description.

    Returns:
        A list of alerts.
    """
    alerts: List[Alert] = []

    alerts += generic_alerts(description)

    if description["type"] == "Unsupported":
        alerts += unsupported_alerts()
    else:
        alerts += supported_alerts(description)

        if description["type"] == "Categorical":
            alerts += categorical_alerts(config, description)
        if description["type"] == "Numeric":
            alerts += numeric_alerts(config, description)
        if description["type"] == "TimeSeries":
            alerts += timeseries_alerts(config, description)
        if description["type"] == "Boolean":
            alerts += boolean_alerts(config, description)

    for idx in range(len(alerts)):
        alerts[idx].column_name = col
        alerts[idx].values = description
    return alerts


def check_correlation_alerts(config: Settings, correlations: dict) -> List[Alert]:
    alerts: List[Alert] = []

    correlations_consolidated = {}
    for corr, matrix in correlations.items():
        if config.correlations[corr].warn_high_correlations:
            threshold = config.correlations[corr].threshold
            correlated_mapping = perform_check_correlation(matrix, threshold)
            for col, fields in correlated_mapping.items():
                set(fields).update(set(correlated_mapping.get(col, [])))
                correlations_consolidated[col] = fields

    if len(correlations_consolidated) > 0:
        for col, fields in correlations_consolidated.items():
            alerts.append(
                HighCorrelationAlert(
                    column_name=col,
                    values={"corr": "overall", "fields": fields},
                )
            )
    return alerts


def get_alerts(
    config: Settings, table_stats: dict, series_description: dict, correlations: dict
) -> List[Alert]:
    alerts: List[Alert] = check_table_alerts(table_stats)
    for col, description in series_description.items():
        alerts += check_variable_alerts(config, col, description)
    alerts += check_correlation_alerts(config, correlations)
    alerts.sort(key=lambda alert: str(alert.alert_type))
    return alerts


def alert_value(value: float) -> bool:
    return not pd.isna(value) and value > 0.01


def skewness_alert(v: float, threshold: int) -> bool:
    return not pd.isna(v) and (v < (-1 * threshold) or v > threshold)


def type_date_alert(series: pd.Series) -> bool:
    from dateutil.parser import ParserError, parse

    try:
        series.apply(parse)
    except ParserError:
        return False
    else:
        return True
