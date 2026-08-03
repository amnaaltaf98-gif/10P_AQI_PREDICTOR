"""
EPA breakpoint conversion utilities.
Open-Meteo and OpenWeather return raw pollutant concentrations, not a single AQI number.
We compute AQI ourselves so historical and live data are calculated the exact same way.
"""

from config import PM25_BREAKPOINTS, PM10_BREAKPOINTS


def _linear_scale(conc, bp_lo, bp_hi, aqi_lo, aqi_hi):
    return ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (conc - bp_lo) + aqi_lo


def concentration_to_aqi(conc, breakpoints):
    """Convert a pollutant concentration to its EPA sub-index using the given breakpoint table."""
    if conc is None:
        return None

    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= conc <= bp_hi:
            return round(_linear_scale(conc, bp_lo, bp_hi, aqi_lo, aqi_hi))

    # above the top breakpoint -> cap at 500
    if conc > breakpoints[-1][1]:
        return 500
    # below the bottom breakpoint (negative reading, sensor noise) -> floor at 0
    return 0


def compute_aqi(pm25=None, pm10=None):
    """
    Overall AQI = the max of the individual pollutant sub-indices.
    This mirrors how the EPA and most public AQI monitors compute the reported number.
    """
    sub_indices = []

    if pm25 is not None:
        sub_indices.append(concentration_to_aqi(pm25, PM25_BREAKPOINTS))
    if pm10 is not None:
        sub_indices.append(concentration_to_aqi(pm10, PM10_BREAKPOINTS))

    sub_indices = [v for v in sub_indices if v is not None]
    if not sub_indices:
        return None

    return max(sub_indices)