from ydata_profiling.config import Settings
from ydata_profiling.report.formatters import (
    fmt,
    fmt_bytesize,
    fmt_monotonic,
    fmt_numeric,
    fmt_percent,
)
from ydata_profiling.report.presentation.core import (
    Container,
    FrequencyTable,
    Image,
    Table,
    VariableInfo,
)
from ydata_profiling.report.structure.variables.render_common import render_common
from ydata_profiling.visualisation.plot import histogram, mini_histogram


def render_real(config: Settings, summary: dict) -> dict:
    varid = summary["varid"]
    template_variables = render_common(config, summary)
    image_format = config.plot.image_format

    name = "Número Real (&Ropf;)"

    # Top
    info = VariableInfo(
        summary["varid"],
        summary["varname"],
        name,
        summary["alerts"],
        summary["description"],
        style=config.html.style,
    )

    table1 = Table(
        [
            {
                "name": "Distintos",
                "value": fmt(summary["n_distinct"]),
                "alert": "n_distinct" in summary["alert_fields"],
            },
            {
                "name": "Distintos (%)",
                "value": fmt_percent(summary["p_distinct"]),
                "alert": "p_distinct" in summary["alert_fields"],
            },
            {
                "name": "Faltantes",
                "value": fmt(summary["n_missing"]),
                "alert": "n_missing" in summary["alert_fields"],
            },
            {
                "name": "Faltantes (%)",
                "value": fmt_percent(summary["p_missing"]),
                "alert": "p_missing" in summary["alert_fields"],
            },
            {
                "name": "Infinito",
                "value": fmt(summary["n_infinite"]),
                "alert": "n_infinite" in summary["alert_fields"],
            },
            {
                "name": "Infinito (%)",
                "value": fmt_percent(summary["p_infinite"]),
                "alert": "p_infinite" in summary["alert_fields"],
            },
            {
                "name": "Média",
                "value": fmt_numeric(
                    summary["mean"], precision=config.report.precision
                ),
                "alert": False,
            },
        ],
        style=config.html.style,
    )

    table2 = Table(
        [
            {
                "name": "Mínimo",
                "value": fmt_numeric(summary["min"], precision=config.report.precision),
                "alert": False,
            },
            {
                "name": "Máximo",
                "value": fmt_numeric(summary["max"], precision=config.report.precision),
                "alert": False,
            },
            {
                "name": "Zeros",
                "value": fmt(summary["n_zeros"]),
                "alert": "n_zeros" in summary["alert_fields"],
            },
            {
                "name": "Zeros (%)",
                "value": fmt_percent(summary["p_zeros"]),
                "alert": "p_zeros" in summary["alert_fields"],
            },
            {
                "name": "Negativo",
                "value": fmt(summary["n_negative"]),
                "alert": False,
            },
            {
                "name": "Negativo (%)",
                "value": fmt_percent(summary["p_negative"]),
                "alert": False,
            },
            {
                "name": "Tamanho em memória",
                "value": fmt_bytesize(summary["memory_size"]),
                "alert": False,
            },
        ],
        style=config.html.style,
    )

    if isinstance(summary.get("histogram", []), list):
        mini_histo = Image(
            mini_histogram(
                config,
                [x[0] for x in summary.get("histogram", [])],
                [x[1] for x in summary.get("histogram", [])],
            ),
            image_format=image_format,
            alt="Mini histograma",
        )
    else:
        mini_histo = Image(
            mini_histogram(config, *summary["histogram"]),
            image_format=image_format,
            alt="Mini histograma",
        )

    template_variables["top"] = Container(
        [info, table1, table2, mini_histo], sequence_type="grid"
    )

    quantile_statistics = Table(
        [
            {
                "name": "Mínimo",
                "value": fmt_numeric(summary["min"], precision=config.report.precision),
            },
            {
                "name": "5-th percentil",
                "value": fmt_numeric(summary["5%"], precision=config.report.precision),
            },
            {
                "name": "Q1",
                "value": fmt_numeric(summary["25%"], precision=config.report.precision),
            },
            {
                "name": "mediana",
                "value": fmt_numeric(summary["50%"], precision=config.report.precision),
            },
            {
                "name": "Q3",
                "value": fmt_numeric(summary["75%"], precision=config.report.precision),
            },
            {
                "name": "95-th percentil",
                "value": fmt_numeric(summary["95%"], precision=config.report.precision),
            },
            {
                "name": "Máximo",
                "value": fmt_numeric(summary["max"], precision=config.report.precision),
            },
            {
                "name": "Intervalo",
                "value": fmt_numeric(
                    summary["range"], precision=config.report.precision
                ),
            },
            {
                "name": "Amplitude interquartil (AIQ)",
                "value": fmt_numeric(summary["iqr"], precision=config.report.precision),
            },
        ],
        name="Estatísticas dos quantis",
        style=config.html.style,
    )

    descriptive_statistics = Table(
        [
            {
                "name": "Desvio padrão",
                "value": fmt_numeric(summary["std"], precision=config.report.precision),
            },
            {
                "name": "Coeficiente de variação (CV)",
                "value": fmt_numeric(summary["cv"], precision=config.report.precision),
            },
            {
                "name": "Curtose",
                "value": fmt_numeric(
                    summary["kurtosis"], precision=config.report.precision
                ),
            },
            {
                "name": "Média",
                "value": fmt_numeric(
                    summary["mean"], precision=config.report.precision
                ),
            },
            {
                "name": "Desvio absoluto mediano (DAM)",
                "value": fmt_numeric(summary["mad"], precision=config.report.precision),
            },
            {
                "name": "Assimetria",
                "value": fmt_numeric(
                    summary["skewness"], precision=config.report.precision
                ),
                "class": "alert" if "skewness" in summary["alert_fields"] else "",
            },
            {
                "name": "Soma",
                "value": fmt_numeric(summary["sum"], precision=config.report.precision),
            },
            {
                "name": "Variância",
                "value": fmt_numeric(
                    summary["variance"], precision=config.report.precision
                ),
            },
            {
                "name": "Monotonicidade",
                "value": fmt_monotonic(summary["monotonic"]),
            },
        ],
        name="Estatística descritiva",
        style=config.html.style,
    )

    statistics = Container(
        [quantile_statistics, descriptive_statistics],
        anchor_id=f"{varid}statistics",
        name="Estatísticas",
        sequence_type="grid",
    )

    if isinstance(summary.get("histogram", []), list):
        hist_data = histogram(
            config,
            [x[0] for x in summary.get("histogram", [])],
            [x[1] for x in summary.get("histogram", [])],
        )
        bins = len(summary["histogram"][0][1]) - 1 if "histogram" in summary else 0
        hist_caption = f"<strong>Histograma com intervalos de tamanho fixo</strong> (bins={bins})"
    else:
        hist_data = histogram(config, *summary["histogram"])
        hist_caption = f"<strong>Histograma com intervalos de tamanho fixos</strong> (bins={len(summary['histogram'][1]) - 1})"

    hist = Image(
        hist_data,
        image_format=image_format,
        alt="Histogram",
        caption=hist_caption,
        name="Histograma",
        anchor_id=f"{varid}histogram",
    )

    fq = FrequencyTable(
        template_variables["freq_table_rows"],
        name="Valores frequentes",
        anchor_id=f"{varid}common_values",
        redact=False,
    )

    evs = Container(
        [
            FrequencyTable(
                template_variables["firstn_expanded"],
                name=f"Mínimo {config.n_extreme_obs} valores",
                anchor_id=f"{varid}firstn",
                redact=False,
            ),
            FrequencyTable(
                template_variables["lastn_expanded"],
                name=f"Máximo {config.n_extreme_obs} valores",
                anchor_id=f"{varid}lastn",
                redact=False,
            ),
        ],
        sequence_type="tabs",
        name="Valores Extremos",
        anchor_id=f"{varid}extreme_values",
    )

    template_variables["bottom"] = Container(
        [statistics, hist, fq, evs],
        sequence_type="tabs",
        anchor_id=f"{varid}bottom",
    )

    return template_variables
