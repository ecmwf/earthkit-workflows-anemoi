from __future__ import annotations

from datetime import datetime
from typing import Any, Generator, TYPE_CHECKING

from earthkit.data import FieldList, SimpleFieldList, ArrayField
from earthkit.data.utils.metadata.dict import UserMetadata
# from earthkit.data.utils.dates import time_to_grib, date_to_grib

from anemoi.utils.dates import frequency_to_seconds

if TYPE_CHECKING:
    from anemoicascade.runner import CascadeRunner

def run(input_state: dict, runner: CascadeRunner, lead_time: int) -> Generator[Any, None, None]:
    """
    Run the model

    Parameters
    ----------
    runner : CascadeRunner
        CascadeRunner object
    input_state : dict
        Initial conditions for the model
    lead_time : int
        Lead time for the model

    Yields
    ------
    Generator[Any, None, None]
        State of the model at each time step
    """
    yield from runner.run(input_state=input_state, lead_time=lead_time)

def run_as_earthkit(input_state: dict, runner: CascadeRunner, lead_time: Any) -> Generator[SimpleFieldList, None, None]:
    """
    Run the model and yield the results as earthkit FieldList

    Parameters
    ----------
    runner : CascadeRunner
        CascadeRunner Object
    input_state : dict
        Initial Conditions for the model
    lead_time : Any
        Lead time for the model

    Yields
    ------
    Generator[SimpleFieldList, None, None]
        State of the model at each time step
    """    
    initial_date: datetime = input_state['date']

    for state in run(input_state, runner, lead_time):
        fields = []
        step = frequency_to_seconds(state['date'] - initial_date) // 3600
        
        for field in state['fields']:
            array = state['fields'][field]
            if '_grib_templates_for_output' in state and field in state['_grib_templates_for_output']:
                metadata = state['_grib_templates_for_output'][field].metadata()
                metadata = metadata.override({'step': step}, headers_only_clone = False) # 'date': time_to_grib(initial_date), 'time': time_to_grib(initial_date)

            else:
                metadata = UserMetadata(
                    {'shortName': field, 'step': step, 'base_datetime': initial_date, "latitudes": runner.checkpoint.latitudes, "longitudes": runner.checkpoint.longitudes}, 
                    array
                )
            fields.append(ArrayField(array, metadata))

        yield FieldList.from_fields(fields)

def collect_as_earthkit(input_state: dict, runner: CascadeRunner, lead_time: Any) -> SimpleFieldList:
    """
    Collect the results of the model run as earthkit FieldList

    Parameters
    ----------
    runner : CascadeRunner
        CascadeRunner object
    input_state : dict
        Initial conditions for the model
    lead_time : Any
        Lead time for the model

    Returns
    -------
    SimpleFieldList
        Combined FieldList of the model run
    """
    fields = []
    for state in run_as_earthkit(input_state, runner, lead_time):
        fields.extend(state.fields)
        
    return SimpleFieldList(fields)