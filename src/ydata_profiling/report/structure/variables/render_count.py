from ydata_profiling.config import Settings
from ydata_profiling.report.formatters import (
    fmt,
    fmt_bytesize,
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


def render_count(config: Settings, summary: dict) -> dict:
    template_variables = render_common(config, summary)
    image_format = config.plot.image_format

    # Top
    info = VariableInfo(
        summary["varid"],
        summary["varname"],
        "Número Real (&Ropf; / &Ropf;<sub>&ge;0</sub>)",
        summary["alerts"],
        summary["description"],
        style=config.html.style,
    )

    table1 = Table(
        [
            {
                "name": "Distintos",
                "value": fmt(summary["n_distinct"]),
                "alert": False,
            },
            {
                "name": "Distintos (%)",
                "value": fmt_percent(summary["p_distinct"]),
                "alert": False,
            },
            {
                "name": "Faltantes",
                "value": fmt(summary["n_missing"]),
                "alert": False,
            },
            {
                "name": "Faltantes (%)",
                "value": fmt_percent(summary["p_missing"]),
                "alert": False,
            },
        ],
        style=config.html.style,
    )

    table2 = Table(
        [
            {
                "name": "Média",
                "value": fmt_numeric(
                    summary["mean"], precision=config.report.precision
                ),
                "alert": False,
            },
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
                "alert": False,
            },
            {
                "name": "Zeros (%)",
                "value": fmt_percent(summary["p_zeros"]),
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

    mini_histo = Image(
        mini_histogram(config, *summary["histogram"]),
        image_format=image_format,
        alt="Mini histograma",
    )

    template_variables["top"] = Container(
        [info, table1, table2, mini_histo], sequence_type="grid"
    )

    seqs = [
        Image(
            histogram(config, *summary["histogram"]),
            image_format=image_format,
            alt="Histogram",
            caption=f"<strong>Histograma com intervalos de tamanho fixo</strong> (bins={len(summary['histogram'][1]) - 1})",
            name="Histograma",
            anchor_id="histogram",
        )
    ]

    fq = FrequencyTable(
        template_variables["freq_table_rows"],
        name="Valores frequentes",
        anchor_id="common_values",
        redact=False,
    )

    evs = Container(
        [
            FrequencyTable(
                template_variables["firstn_expanded"],
                name=f"{config.n_extreme_obs} menores valores",
                anchor_id="firstn",
                redact=False,
            ),
            FrequencyTable(
                template_variables["lastn_expanded"],
                name=f"{config.n_extreme_obs} maiores valores",
                anchor_id="lastn",
                redact=False,
            ),
        ],
        sequence_type="tabs",
        name="Valores Extremos",
        anchor_id="extreme_values",
    )

    template_variables["bottom"] = Container(
        [
            Container(
                seqs, sequence_type="tabs", name="Histograma(s)", anchor_id="histograms"
            ),
            fq,
            evs,
        ],
        sequence_type="tabs",
        anchor_id=summary["varid"],
    )

    return template_variables
